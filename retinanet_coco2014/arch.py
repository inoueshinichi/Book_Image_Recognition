"""
RetinaNet with MS_COCO 2014
"""
import os
import sys
import glob
from typing import (
    Callable,
    Sequence,
    Tuple,
    Union,
    List,
    Dict,
    Optional,
)
import json
import math
import shutil
from pathlib import Path
import random
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment # ハンガリアンアルゴリズム
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm

# torch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim

# torchvision
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import (
    draw_bounding_boxes,
    draw_keypoints,
    draw_segmentation_masks,
)
from torchvision.ops import (
    sigmoid_focal_loss,
    batched_nms,
)
from torchvision.ops.misc import FrozenBatchNorm2d

# MS_COCO
from pycocotools.cocoeval import COCOeval

# My libs
from utils import (
    post_process,
)


"""NNモジュール"""

class FrozenResidualBlock(nn.Module):
    """
    ResNet18における残差ブロック
    in_channels : 入力チャネル数
    out_channels: 出力チャネル数
    stride      : 畳み込み層のストライド
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = FrozenBatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # strideが1より大きいときにスキップ接続と残差接続の高さと幅を
        # 合わせるため、別途畳み込み演算を用意
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                FrozenBatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor):
        """

        :param x: (B, C, H, W)
        :return:
        """

        # 残差接続
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        # 残差ブロックの出力とスキップブロックの出力を合流
        out += x

        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet 18
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = FrozenBatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)

        self.layer1 = nn.Sequential(
            FrozenResidualBlock(64, 64),
            FrozenResidualBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            FrozenResidualBlock(64, 128, stride=2),
            FrozenResidualBlock(128, 128),
        )

        self.layer3 = nn.Sequential(
            FrozenResidualBlock(128, 256, stride=2),
            FrozenResidualBlock(256, 256),
        )

        self.layer4 = nn.Sequential(
            FrozenResidualBlock(256, 512, stride=2),
            FrozenResidualBlock(512, 512),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


class FeaturePyramidNetwork(nn.Module):
    """
    特徴ピラミッドネットワーク
    num_features: 出力特徴量のチャネル
    """
    def __init__(self, num_features: int = 256):
        super().__init__()

        # 特徴ピラミッドネットワークから出力される階層レベル
        # バックボーンネットワークの最終層の特徴マップを第5階層とし、
        # 縮小方向に第6, 7階層の2つの特徴マップを、
        # 拡大方向に第3, 4階層の2つの特徴マップを生成
        self.levels = (3, 4, 5, 6, 7)

        ''' 縮小方向の特徴抽出 '''
        self.p6 = nn.Conv2d(512,
                            num_features,
                            kernel_size=3,
                            stride=2,
                            padding=1)
        self.p7_relu = nn.ReLU(inplace=True)
        self.p7 = nn.Conv2d(num_features,
                            num_features,
                            kernel_size=3,
                            stride=2,
                            padding=1)

        ''' 拡大方向の特徴抽出 '''
        self.p5_1 = nn.Conv2d(512,
                              num_features,
                              kernel_size=1)
        self.p5_2 = nn.Conv2d(num_features,
                              num_features,
                              kernel_size=3,
                              padding=1)
        self.p4_1 = nn.Conv2d(256,
                              num_features,
                              kernel_size=1)
        self.p4_2 = nn.Conv2d(num_features,
                              num_features,
                              kernel_size=3,
                              padding=1)
        self.p3_1 = nn.Conv2d(128,
                              num_features,
                              kernel_size=1)
        self.p3_2 = nn.Conv2d(num_features,
                              num_features,
                              kernel_size=3,
                              padding=1)


    def forward(self,
                c3: torch.Tensor,
                c4: torch.Tensor,
                c5: torch.Tensor):
        """

        :param c3: ResNet18 c3特徴マップ (B, C, H, W)
        :param c4: ResNet18 c4特徴マップ (B, C, H, W)
        :param c5: ResNet18 c5特徴マップ (B, C, H, W)
        :return:
        """

        ''' 縮小方向の特徴抽出 '''
        p6 = self.p6(c5)
        p7 = self.p7_relu(p6)
        p7 = self.p7(p7)

        ''' 拡大方向の特徴抽出 '''
        p5 = self.p5_1(c5)
        p5_up = F.interpolate(p5, scale_factor=2)
        p5 = self.p5_2(p5)

        p4 = self.p4_1(c4) + p5_up
        p4_up = F.interpolate(p4, scale_factor=2)
        p4 = self.p4_2(p4)

        p3 = self.p3_1(c3) + p4_up
        p3 = self.p3_2(p3)

        return p3, p4, p5, p6, p7


class AnchorBoxGenerator:
    """
    予測検出矩形の基準となるアンカーボックスを生成するクラス
    levels: 入力特徴マップの階層レベル
    """
    def __init__(self, levels: int):

        # 用意するアンカーボックスのアスペクト比
        ratios: torch.Tensor = torch.tensor([0.5, 1.0, 2.0])

        # 用意するアンカーボックスの基準となる大きさに対するスケール
        scales = torch.tensor([2**0, 2**(1/3), 2**(2/3)])

        # 1つのアスペクト比に対して全スケールのアンカーボックスを
        # 用意するのでアンカーボックスの数=アスペクト比の数 * スケール数
        self.num_anchors = ratios.shape[0] * scales.shape[0]

        # 各階層の特徴マップでの1画素の移動量が入力画像での何画素の移動になるかを示す値
        # 2**Nのスケールで縮小するので, 1画素の移動量入力画像では, 2**N倍される
        self.strides = [2 ** level for level in levels]

        self.anchors = []
        for level in levels:
            # 現階層における基準となる正方形のアンカーボックスの1辺の長さ
            # 深い階層のアンカーボックスには大きい物体の
            # 検出を担当させるため, 基準の長さを長く設定
            base_length = 2 ** (level + 2)
            # 0: 2**2 = 4
            # 1: 2**3 = 8
            # 2: 2**4 = 16
            # 3: 2**5 = 32
            # 4: 2**6 = 64
            # 5: 2**7 = 128
            # 6: 2**8 = 256
            # 7: 2**9 = 512

            # アンカーボックスの1辺の長さをスケール
            scaled_lengths = base_length * scales
            # アンカーボックスが正方形の場合の面積を計算
            anchor_areas = scaled_lengths ** 2

            # アスペクト比(ratio=height/ratio)に応じて辺の長さを変更
            # width * height = width * (width * ratio) = area
            # width = (area / ratio) ** 0.5
            # unsqueezeとブロードキャストにより
            # アスペクト比 * スケール数の数のアンカーボックスの幅と高さを生成
            # (3,) * (3,1) = (3, 3)
            # e.g
            # a = [1, 3.5, 6]
            # b = [[0.5], [1], [2]]
            # a*b = [
            #  [ 0.5, 1.75, 3],
            #  [ 1, 3.5, 6],
            #  [ 2, 7, 12]
            # ]
            anchor_widths = (anchor_areas / ratios.unsqueeze(1)) ** 0.5  # (3,)*(3,1)=(3,3)
            anchor_heights = anchor_widths * ratios.unsqueeze(1)  # (3,3)

            # (3,3) -> (9,)
            anchor_widths = anchor_widths.flatten()
            anchor_heights = anchor_heights.flatten()

            # アンカーボックスの中心を原点(0,0)としたときの
            # x_min, y_min, x_max, y_maxのオフセット
            anchor_x_mins = - 0.5 * anchor_widths  # (9,)
            anchor_y_mins = - 0.5 * anchor_heights  # (9,)
            anchor_x_maxs = 0.5 * anchor_widths  # (9,)
            anchor_y_maxs = 0.5 * anchor_heights  # (9,)

            level_anchors = torch.stack(
                (anchor_x_mins, anchor_y_mins, anchor_x_maxs, anchor_y_maxs),
                dim=1)  # (9,4)

            self.anchors.append(level_anchors)  # [(9,4),(9,4),(9,4),(9,4),(9,4)]

    @torch.no_grad()
    def generate(self, feature_sizes: List[torch.Size]):
        """
        アンカーボックスの生成
        :param features_sizes: 入力される特徴マップのそれぞれの大きさ
        :return:
        """
        anchors = []
        # stride: (L,1)
        # level_anchors: (L,9,4)
        # feature_size: (L, H_p[i]],W_p[i]) i= 1,...,L
        for stride, level_anchors, feature_size in zip(self.strides, self.anchors, feature_sizes):
            # 現階層の特徴マップの大きさ
            height, width = feature_size

            # 入力画像の画素の移動量を表すstridesを使って
            # 特徴マップの画素の位置 -> 入力画僧の画素の位置に変換
            # (画像の中心位置を計算するために0.5を加算)
            # x_at_in = 2^l * (x + 0.5), y_at_in = 2^l * (y + 0.5)
            xs = (torch.arange(width) + 0.5) * stride  # 入力画像上
            ys = (torch.arange(height) + 0.5) * stride  # 入力画像上

            # 入力画像座標上のグリッド(x,y)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')

            grid_x = grid_x.flatten()  # (W_p[i]*H_p[i]),)
            grid_y = grid_y.flatten()  # (H_p[i]*W_p[i],)

            # 各画像の中心位置にアンカーボックスの
            # x_min,y_min,x_max,y_maxのオフセットを加算
            anchor_x_mins = (grid_x.unsqueeze(1) + level_anchors[:, 0]).flatten()
            anchor_y_mins = (grid_y.unsqueeze(1) + level_anchors[:, 1]).flatten()
            anchor_x_maxs = (grid_x.unsqueeze(1) + level_anchors[:, 2]).flatten()
            anchor_y_maxs = (grid_y.unsqueeze(1) + level_anchors[:, 3]).flatten()

            # 第1軸を追加してx_min, y_min, x_max, y_maxを連結
            level_anchors = torch.stack(
                (anchor_x_mins, anchor_y_mins, anchor_x_maxs, anchor_y_maxs),
                dim=1
            )  # (H_p[i]*W_p[i]*9, 4)
            anchors.append(level_anchors)  # list[(H_p[i]*W_p[i]*9, 4), ...] = (L,(H_p[i]*W_p[i]*9, 4)

        # 全階層のアンカーボックスを連結
        anchors = torch.cat(anchors, dim=0)  # (sum(H_p[i]*W_p[i]*9), 4)

        return anchors


class DetectionHead(nn.Module):
    """
    検出ヘッド(分類や矩形の回帰に使用)
    num_channels_per_anchor: 1アンカーに必要な出力チャネル数
    num_anchors            : アンカー数
    num_features           : 入力および中間特徴量のチャネル数
    """

    def __init__(self,
                 num_channels_per_anchor: int,
                 num_anchors: int = 9,
                 num_features: int = 256):
        super().__init__()

        self.num_anchors = num_anchors

        # 特徴ピラミッドネットワークの特徴マップを分類や回帰専用の
        # 特徴マップに変換するための畳み込みブロック
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(num_features, num_features,
                                    kernel_size=3, padding=1),
                          nn.ReLU(inplace=True))
            for _ in range(4)])

        # 検出ヘッドの出力チャネル数を設定する
        # 分類ヘッドに使用する場合, アンカーボックス数 x 物体クラス数
        # 矩形ヘッドに使用する場合, アンカーボックス数 x 4 (cx,cy,w,h)
        self.out_conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_anchors * num_channels_per_anchor,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # (K,S,P) = (3,1,1) -> half

    def forward(self, x: torch.Tensor):
        for i in range(4):
            x = self.conv_blocks[i](x)
        x = self.out_conv(x)

        bs, c, h, w = x.shape

        # 後処理に備えて予測結果(検出矩形)の並び替え
        # (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0,2,3,1)
        # 第1軸に全画素の予測結果を並べる
        x = x.reshape(bs, w*h*self.num_anchors, -1)

        return x


class RetinaNet(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()

        self.backbone = ResNet18()

        self.fpn = FeaturePyramidNetwork()

        self.anchor_generator = AnchorBoxGenerator(self.fpn.levels)

        # 分類ヘッド\矩形ヘッド
        # 検出ヘッドは全ての特徴マップで共有
        self.class_head = DetectionHead(
            num_channels_per_anchor=num_class,
            num_anchors=self.anchor_generator.num_anchors
        )
        self.box_head = DetectionHead(
            num_channels_per_anchor=4,
            num_anchors=self.anchor_generator.num_anchors
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

        # 分類ヘッドの出力にシグモイドを適用して各クラスの確率を出力
        # 学習開始時の確率が0.01になるようにパラメータを初期化
        prior = torch.tensor(0.01)
        nn.init.zeros_(self.class_head.out_conv.weight)
        nn.init.constant_(self.class_head.out_conv.bias,
                          -((1.0 - prior) / prior).log())

        # 学習開始時のアンカーボックスの中心位置の移動が0,
        # 大きさが1倍となるように矩形ヘッドを初期化
        nn.init.zeros_(self.box_head.out_conv.weight)
        nn.init.zeros_(self.box_head.out_conv.bias)

    def get_device(self):
        return self.backbone.conv1.weight.device

    def forward(self, x: torch.Tensor):
        """

        :param x: (B, C, H, W)
        :return:
        """
        cs = self.backbone(x)
        ps = self.fpn(*cs) # p3,p4,p5,p6,p7

        # p3,p4,p5,p6,p7に対して
        # 分類ヘッドと矩形ヘッドを適用(パラメータ共有)
        class_head_out_list = list(map(self.class_head, ps))
        box_head_out_list = list(map(self.box_head, ps))

        '''各特徴量マップに対するヘッドの結果を連結'''
        # [(B, H_p3*W_p3*anchors, classes), ..., (B, H_p7*W_p7*anchors, classes)]
        preds_class = torch.cat(class_head_out_list, dim=1)  # (B, sum(H_p[i]*W_p[i]*anchor)), classes)
        preds_box = torch.cat(box_head_out_list, dim=1)  # (B, sum(H_p[i]*W_p[i]*anchor)), 4)

        '''アンカーボックスを生成'''
        feature_sizes = [p.shape[2:] for p in ps]  # [(H_p3,W_p3),(H_p4,W_p4),(H_p5,W_p5),(H_p6,W_p6),(H_p7,W_p7)]
        anchors = self.anchor_generator.generate(feature_sizes)  # (levels=5, H_p[i]*W_p[i]*9, 4)
        anchors = anchors.to(x.device)  # cpu -> cuda

        return preds_class, preds_box, anchors
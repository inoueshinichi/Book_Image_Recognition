"""
DETR (DEtection TRansformer) with MS_COCO 2014
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


"""NNモジュール"""

class PositionalEncoding:
    """
    位置エンコーディング生成クラス
    eps : 0割防止の小さい値
    temperature: 温度定数
    """
    def __init__(self,
                 eps: float = 1e-6,
                 temperature: int = 10000,
                 ):
        self.eps = eps
        self.temperature = temperature


    @torch.no_grad()
    def __call__(self, x: torch.Tensor,
                 mask: torch.Tensor):
        """
        位置エンコーディングを生成する関数
        :param x: 特徴マップ [B, C, H, W]
        :param mask: 画像領域を示すマスク [B, H, W] Batchの各値は画像サイズが異なるため. 画像領域: false, 非画像領域: true
        :return:
        """
        # 位置エンコーディングのチャネル数は入力の半分として
        # x方向のエンコーディングとy方向のエンコーディングを用意して
        # それらを連結することで入力のチャネル数に合わせる
        num_pos_channels = x.shape[1] // 2

        # 温度定数の指数
        dim_t = torch.arange(0, num_pos_channels, 2,
                             dtype=x.dtype, device=x.device)

        # sinとcosを計算するために値を複製
        # [0, 2, ...] -> [0,0,2,2...]
        dim_t = dim_t.repeat_interleave(2)

        # sinとcosへの入力の分母となるT^{2i / d}を計算
        dim_t /= num_pos_channels
        dim_t = self.temperature ** dim_t  # (C,)

        # マスクされていない領域の座標を計算
        inverted_mask = ~mask
        y_encoding = inverted_mask.cumsum(1, dtype=torch.float32)  # (B, H dim=1, W)
        x_encoding = inverted_mask.cumsum(2, dtype=torch.float32)  # (B, H, W dim=2)

        # 座標を0-1に正規化して2PIを掛ける
        y_encoding = 2 * math.pi * y_encoding / (y_encoding.max(dim=1, keepdim=True)[0] + self.eps)
        x_encoding = 2 * math.pi * x_encoding / (x_encoding.max(dim=2, keepdim=True)[0] + self.eps)

        # 座標を保持するテンソルにチャネル軸を追加して
        # チャネル軸方向にdim_tで割る
        # 偶数チャネルはsin 奇数チャネルはcosの位置エンコーディングになる
        y_encoding = y_encoding.unsqueeze(dim=1) / dim_t.view(num_pos_channels, 1, 1) # (B, H, W) => (B, C, H, W)

        y_encoding[:, ::2] = y_encoding[:, ::2].sin()
        y_encoding[:, 1::2] = y_encoding[:, 1::2].cos()

        x_encoding = x_encoding.unsqueeze(dim=1) / dim_t.view(num_pos_channels, 1, 1) # (B, C, H, W)
        x_encoding[:, ::2] = x_encoding[:, ::2].sin()
        x_encoding[:, 1::2] = x_encoding[:, 1::2].cos()

        # 位置エンコーディングを連結して出力
        return torch.cat([x_encoding, y_encoding], dim=1)


class TransformerEncoderLayer(nn.Module):
    """
    DETRのエンコーダ層
    dim_hidden: 特徴量の次元
    num_heads: MHSAのヘッド数
    dim_feedforward: FNNの中間特徴量の次元
    dropout: ドロップアウト率
    """
    def __init__(self,
                 dim_hidden: int = 256,
                 num_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # 自己アテンションブロック
        self.self_attn = nn.MultiheadAttention(dim_hidden, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_hidden)

        # FNN
        self.fnn = nn.Sequential(
            nn.Linear(dim_hidden, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_hidden)
        )

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_hidden)

    def forward(self,
                x: torch.Tensor,
                pos_encoding: torch.Tensor,
                mask: torch.Tensor):
        """
        L: 特徴量数
        B: バッチサイズ
        D: 特徴量の次元
        ---
        :param x:  [L, B, D]
        :param pos_encoding: [L, B, D]
        :param mask: [B, L]
        :return:
        """

        # クエリとキーに位置エンコーディングを加算
        q = k = x + pos_encoding

        # self_attnにはクエリ、キー、バリューの順番に入力
        # key_padding_maskにmaskを渡すことでマスクが真の値を持つ領域の
        # キーは使われなくなり、特徴収集の対象から外れる
        # MutltiheadAttentionクラスは特徴収集結果とアテンションの値の
        # 2つの結果を返す
        x_attn, attn_map = self.self_attn(q, k, x,
                                          key_padding_mask=mask)  # mask: 有効値: False, 無効値: True
        x = x + self.dropout1(x_attn)
        x = self.norm1(x)

        x_fnn  = self.fnn(x)
        x = x + self.dropout2(x_fnn)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    DETRのデーコーダ層
    dim_hidden: 特徴量の次元
    num_heads: MHSAのヘッド数
    dim_feedforward: FNNの中間特徴量の次元
    dropout: ドロップアウト率
    """
    def __init__(self,
                 dim_hidden: int = 256,
                 num_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        # 自己アテンションブロック
        self.self_attn = nn.MultiheadAttention(dim_hidden, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_hidden)

        # 物体特徴量と特徴マップの特徴量の交差アテンション
        self.crs_attn = nn.MultiheadAttention(dim_hidden, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_hidden)

        # FNN
        self.fnn = nn.Sequential(
            nn.Linear(dim_hidden, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_hidden)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim_hidden)

    def forward(self,
                h: torch.Tensor,
                query_embed: torch.Tensor,
                x: torch.Tensor,
                pos_encoding: torch.Tensor,
                mask: torch.Tensor):
        """

        :param h: 物体特徴量 [Q, B, D]
        :param query_embed: 物体クエリ埋め込み [Q, B, D]
        :param x: 特徴マップの特徴量 [L, B, D]
        :param pos_encoding:  位置エンコーディング [L, B, D]
        :param mask: マスク [B, L]
        :return:
        """
        # 物体クエリ埋め込みの自己アテンション
        # (物体特徴量の自己アテンション)
        q = k = h + query_embed
        x_attn, attn_map = self.self_attn(q, h, k)
        h = h + self.dropout1(x_attn)
        h = self.norm1(h)

        # 物体クエリ埋め込みと特徴マップの交差アテンション
        # (物体特徴量と特徴マップの交差アテンション)
        q = k = h + query_embed
        x_crs_attn, attn_crs_map = self.crs_attn(h + query_embed,  # Q
                                                 x + pos_encoding, # K
                                                 x,  # V
                                                 key_padding_mask=mask)  # key_padding_maskは, cross_attention用のマスク

        h = h + self.dropout2(x_crs_attn)
        h = self.norm2(h)

        h2 = self.fnn(h)
        h = h + self.dropout3(h2)
        h = self.norm3(h)

        return h


class Transformer(nn.Module):
    """
    DETRのTransformerブロック
    dim_hidden: 特徴量の次元
    num_heads: MHSAのヘッド数
    num_encoder_layers: Transformerエンコーダ層の数
    num_decoder_layers: Transformerデコーダ層の数
    dim_feedforward: FNNの中間特徴量の次元
    dropout: ドロップアウト率
    """
    def __init__(self,
                 dim_hidden: int = 256,
                 num_heads: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        # 複数のエンコーダ
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(dim_hidden,
                                    num_heads,
                                    dim_feedforward,
                                    dropout) for _ in range(num_encoder_layers)])

        # 複数のデコーダ
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(dim_hidden,
                                    num_heads,
                                    dim_feedforward,
                                    dropout) for _ in range(num_decoder_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavierの初期化
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self,
                x: torch.Tensor,
                pos_encoding: torch.Tensor,
                mask: torch.Tensor,
                query_embed: torch.Tensor):
        """
        Q: クエリ数
        L: 特徴量数
        B: バッチサイズ
        :param x: 特徴マップ [B, C, H, W]
        :param pos_encoding: 位置エンコーディング [B, C, H, W]
        :param mask: [B, H, W]
        :param query_embed: 物体クエリ埋め込み [Q, D]
        :return:
        """
        batch_size = x.shape[0]

        '''
        エンコーダやデコーダのMHSAのクエリ, キー, バリューは
        第0軸: 特徴量列軸
        第1軸: バッチ軸
        第2軸: チャネル軸
        [L, B, C]

        マスクは
        第0軸: バッチ軸
        第1軸: 特徴量列軸
        [B, L]

        である必要がある
        '''

        # 特徴マップ
        # (B,C,H,W) -> (H*W=L,B,C)
        x = x.flatten(2).permute(2,0,1)

        # 位置エンコーディング
        # (B,C,H,W) -> (H*W=L,B,C)
        pos_encoding = pos_encoding.flatten(2).permute(2,0,1)

        # マスク
        # (B,H,W) -> (B,H*W=L)
        mask = mask.flatten(1)

        # 物体クエリ
        # (Q,class+1) -> (Q,B,class+1)
        query_embed = query_embed.unsqueeze(1).expand(-1,batch_size,-1)

        '''
        各データをエンコーダ層とデコーダ層に入力するフェーズ
        '''
        # エンコーダ層は
        # 特徴量マップxをより良い特徴量に変換する
        for layer in self.encoder_layers:
            x = layer(x, pos_encoding, mask)


        # 途中のデコーダ層の出力を保持
        hs = []
        h = torch.zeros_like(query_embed)  # 初期値

        # デコーダ層は
        # エンコーダ層で得られた特徴量と特徴マップの
        # 交差アテンションで特徴マップから物体特徴量を抽出
        for layer in self.decoder_layers:
            h = layer(h, query_embed, x, pos_encoding, mask)
            hs.append(h)

        # デコーダ層の出力を第0軸で連結
        # Decoders * (Q,B,class+1) -> (Decoders,B,class+1)
        hs = torch.stack(hs, dim=0)
        hs = hs.permute(0,2,1,3) # (Decoders,B,Q,class+1)

        return hs


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


class DETR(nn.Module):
    """
    DETR
    num_queires: 物体クエリ埋め込みの数
    dim_hidden: Transformerで処理する際の特徴量次元
    num_heads: MHSAのヘッド数
    num_encoder_layers: Transformerエンコーダ層の数
    num_decoder_layers: Transformerデコーダ層の数
    dim_feedforward: FNNの中間特徴量の次元
    dropout: ドロップアウト率
    num_class: 物体クラス数
    """

    def __init__(self,
                 num_queries: int,
                 dim_hidden: int,
                 num_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 num_class: int):
        super().__init__()

        self.backbone = ResNet18()

        # バックボーンネットワークの特徴マップのチャネル数を
        # 減らすための畳み込み層
        self.proj = nn.Conv2d(512, dim_hidden, kernel_size=1)

        # Transformerブロック
        self.transformer = Transformer(dim_hidden,
                                       num_heads,
                                       num_encoder_layers,
                                       num_decoder_layers,
                                       dim_feedforward,
                                       dropout)

        # 分類ヘッド
        self.class_head = nn.Linear(dim_hidden, num_class + 1)  # 背景クラスを追加

        # 矩形ヘッド
        self.box_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, 4)
        )

        # 位置エンコーディング
        self.positional_encoding = PositionalEncoding()

        # 物体クラス埋め込み
        self.query_embed = nn.Embedding(num_queries, dim_hidden)

    def get_device(self):
        return self.backbone.conv1.weight.device

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor):
        """

        :param x: 入力画像 (B,C,H,W)
        :param mask: マスク (B,H,W) True: 有効, False: 無効
        :return:
        """
        # バックボーンネットワークから第5レイヤーの特徴マップを取得
        x = self.backbone(x)[-1]  # c3, c4, c5

        # チャネル数を減らす for Transformer
        x = self.proj(x)

        # 入力画像と同じ大きさを持つmaskを特徴マップの大きさにリサイズ
        # interpolate関数はbool型には対応していないため, 一旦, xと同じ型に変換
        mask = mask.to(x.dtype)
        mask = F.interpolate(mask.unsqueeze(1), size=x.shape[2:])[:, 0]  # (B,C,W) -> (B,1,*,*)
        mask = mask.to(torch.bool)

        # 位置エンコーディング
        pos_encoding = self.positional_encoding(x, mask)

        # (Decoders,B,Q,class+1)
        hs = self.transformer(x, pos_encoding, mask, self.query_embed.weight)

        # 分類ヘッド, 矩形ヘッド
        preds_class = self.class_head(hs)
        preds_box = self.box_head(hs).sigmoid() # 論理座標系での出力

        # (Decoders B,Q,class+1), (Decoders, B,Q,4)
        return preds_class, preds_box
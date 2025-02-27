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
    calc_iou,
    convert_to_xyxy,
    convert_to_xywh,
)

"""損失"""

def loss_func(preds_class: torch.Tensor,
              preds_box: torch.Tensor,
              anchors: torch.Tensor,
              targets: dict,
              iou_lower_threshold: float = 0.4,
              iou_upper_threshold: float = 0.5):

    """
    :param preds_class: 物体検出のクラス (B, sum(H_p[i]*W_p[i]*anchor)=アンカーボックス数, classes)
    :param preds_box: 検出矩形のアンカーボックスからの誤差 (B, sum(H_p[i]*W_p[i]*anchor)=アンカーボックス数, 4)
    :param anchors: アンカーボックス (sum(H_p[i]*W_p[i]*9)=アンカーボックス数, 4)
    :param targets: ラベル list[target(dict), ...]
    :param iou_lower_threshold: 検出矩形と正解矩形をマッチさせるか決める下限値
    :param iou_upper_threshold: 検出矩形と正解矩形をマッチさせるか決める上限値
    :return:
    """

    anchors_xywh = convert_to_xywh(anchors)

    # 画像毎に目的関数を計算
    loss_class = preds_class.new_tensor(0)
    loss_box = preds_class.new_tensor(0)
    for img_preds_class, img_preds_box, img_targets in zip(preds_class,
                                                           preds_box,
                                                           targets):

        # 1) 現在の画像に対する正解がないとき
        if img_targets['classes'].shape[0] == 0:
            # 全ての物体クラスの確率が0となるように
            # (背景クラスとして分類されるように)ラベルを作成
            targets_class = torch.zeros_like(img_preds_class)
            # https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html#torchvision.ops.sigmoid_focal_loss
            loss_class += sigmoid_focal_loss(img_preds_class, targets_class, reduction='sum')

            continue

        # 各画素のアンカーボックスと正解矩形のIoUを計算し,
        # 各アンカーボックスに対して最大のIoUを持つ正解矩形を抽出
        ious, _ = calc_iou(anchors, img_targets['boxes']) # アンカーボックス数, 正解矩形数

        # ious: (アンカーボックス数, 正解矩形数)
        # 各アンカーボックスに対して最も大きなIoUを与える正解矩形のインデックスとそのIoUを出力
        ious_max, ious_argmax = ious.max(dim=1)  # (anchors, 1), (anchors, 1)


        # 分類ラベルを-1で初期化
        # IoUが下限値と上限値にあるアンカーボックスは
        # ラベルを-1として損失を計算しないようにする
        targets_class = torch.full_like(img_preds_class, -1)  # (anchors, classes)
        '''
        [-1,-1,-1,...,-1]
        [-1,-1,-1,...,-1]
        [-1,-1,-1,...,-1]
        [-1,-1,-1,....-1]
        [-1,-1,-1,...,-1]
        ...
        [-1,-1,-1,...,-1]
        '''

        # 2) IoUが下限値以下は, 背景(確率0)=[0,...,0]とする
        # アンカーボックスとマッチした正解矩形のIoUが下の閾値より
        # 小さい場合、全ての物体クラスの確率が0となるようラベルを用意
        targets_class[ious_max < iou_lower_threshold] = 0 # (anchors, [0,0,0,...,0])
        '''
        [-1,-1,-1,...,-1]
        [-1,-1,-1,...,-1]
        [0,0,0,...,0] ious_max < iou_lower_threshold
        [0,0,0,....0] ious_max < iou_lower_threshold
        [-1,-1,-1,...,-1]
        ...
        [-1,-1,-1,...,-1]
        '''

        # 3) アンカーボックスとマッチした正解矩形のIoUが上の閾値より
        # 大きい場合、陽性のアンカーボックスとして分類回帰の対象にする
        positive_masks = ious_max > iou_upper_threshold
        num_positive_anchors = positive_masks.sum() # 陽性アンカーボックスの数

        # 陽性のアンカーボックスについて、マッチした正解矩形が示す
        # 物体クラスの確率を1, それ以外を0にする. e.g. [0,0,1,...,0]
        targets_class[positive_masks] = 0  # クラス確率を初期化
        '''
        [0,0,0,...,0] ious_max > iou_upper_threshold
        [-1,-1,-1,...,-1]
        [0,0,0,...,0] ious_max < iou_lower_threshold
        [0,0,0,....0] ious_max < iou_lower_threshold
        [-1,-1,-1,...,-1]
        ...
        [0,0,0,...,0] ious_max > iou_upper_threshold
        '''

        # 各anchorに最も大きなIoUを取る正解矩形のインデックスを割り当てる
        assigned_classes = img_targets['classes'][ious_argmax] # (正解矩形数,) indexing with (anchors,) -> (anchors,)
        # positive_masksでassigned_classesから有効な正解矩形のインデックスを取り出す
        targets_class[positive_masks, assigned_classes[positive_masks]] = 1  # クラスラベルの列に1を立てる
        '''
        [0,0,0,...,1] ious_max > iou_upper_threshold (True)
        [-1,-1,-1,...,-1]
        [0,0,0,...,0] ious_max < iou_lower_threshold
        [0,0,0,....0] ious_max < iou_lower_threshold
        [-1,-1,-1,...,-1]
        ...
        [0,1,0,...,0] ious_max > iou_upper_threshold (True)
        '''

        # 4) IoUが下限値と上限値の間にある(クラス割り当てが不明瞭)な
        # アンカーボックスについては, 分類の損失計算を行わない
        targets_masks = targets_class != -1
        '''
        targets_class = np.array([[0,0,1], [-1,-1,-1], [1,0,0]])
        targets_class
        Out[23]: 
        array([[ 0,  0,  1],
            [-1, -1, -1],
            [ 1,  0,  0]])
        targets_class != -1
        Out[24]: 
        array([[ True,  True,  True],
            [False, False, False],
            [ True,  True,  True]])
        '''
        valid_losses = targets_masks * sigmoid_focal_loss(img_preds_class, targets_class)
        # ここでは, num_positive_anchors == 0のケースがあるので, 0割エラーを回避する.
        # この場合, valid_lossesは全て0の多次元配列なので, valid_losses.sum()もゼロ
        loss_class += valid_losses.sum() / num_positive_anchors.clamp(min=1)

        # 陽性のアンカーボックスが一つも存在しないとき
        # 矩形の誤差の学習はしない
        if num_positive_anchors == 0:
            continue

        # 各アンカーボックスにマッチした正解矩形を抽出
        assigned_boxes = img_targets['boxes'][ious_argmax] # (アンカーボックス数, 正解矩形の4(min_x,min_y,max_x,max_y))
        assigned_boxes_xywh = convert_to_xywh(assigned_boxes)

        # アンカーボックスとマッチした正解矩形との誤差を計算し、ラベルを作成
        targets_box = torch.zeros_like(img_preds_box) # (sum(H_p[i]*W_p[i]*anchor)=アンカーボックス数, 4)
        # 中心位置の誤差はアンカーボックスの大きさでスケール
        targets_box[:, :2] = (assigned_boxes_xywh[:, :2] - anchors_xywh[:, :2]) / anchors_xywh[:, 2:]
        # 大きさはアンカーボックスに対するスケールのlogを予測
        targets_box[:, 2:] = (assigned_boxes_xywh[:, 2:] / anchors_xywh[:, 2:]).log()

        # L1誤差とL2誤差を組み合わせたsmooth L1誤差を使用
        loss_box += F.smooth_l1_loss(img_preds_box[positive_masks],
                                     targets_box[positive_masks],
                                     beta=1 / 9)

    batch_size = preds_class.shape[0]
    loss_class = loss_class / batch_size
    loss_box = loss_box / batch_size

    return loss_class, loss_box
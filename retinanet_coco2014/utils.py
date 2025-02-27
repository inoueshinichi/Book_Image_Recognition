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


"""ヘルパー関数"""

def convert_to_xywh(boxes: torch.Tensor):
    """
    矩形をmin_x, min_y, max_x, max_yからx, y, width, heightに変換する関数
    :param boxes: 矩形集合, [矩形数 (任意の軸数), 4 (min_x, min_y, max_x, max_y)]
    :return:
    """
    wh = boxes[..., 2:] - boxes[..., :2]
    xy = boxes[..., :2] + wh / 2
    boxes = torch.cat((xy, wh), dim=-1)

    return boxes

def convert_to_xyxy(boxes: torch.Tensor):
    """
    矩形をx, y, width, heightからmin_x, min_y, max_x, max_xに変換
    :param boxes: 外接集合, [矩形数 (任意の軸数), 4 (x, y, width, height)]
    :return:
    """
    min_xy = boxes[..., :2] - boxes[..., 2:] / 2
    max_xy = boxes[..., 2:] + min_xy
    boxes = torch.cat((min_xy, max_xy), dim=-1)

    return boxes

def calc_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    第1軸をunsqueezeし、ブロードキャストを利用することで
    [矩形数, 1, 2] と[矩形数, 2]の演算結果が[boxes1の矩形数, boxes2の矩形数, 2] となる
    :param boxes1: 矩形集合, [矩形数, 4 (min_x, min_y, max_x, max_y)]
    :param boxes2: 矩形集合, [矩形数, 4 (min_x, min_y, max_x, max_y)]
    :return:
    """
    # 積集合の左上の座標を取得
    intersect_left_top = torch.maximum(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])
    # 積集合の右下の座標を取得
    intersect_right_bottom = torch.minimum(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])

    # 積集合の幅と高さを算出し、面積を計算
    intersect_width_height = (intersect_right_bottom - intersect_left_top).clamp(min=0)
    intersect_areas = intersect_width_height.prod(dim=2)

    # それぞれの矩形の面積を計算
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 和集合の面積を計算
    union_areas = areas1.unsqueeze(1) + areas2 - intersect_areas

    ious = intersect_areas / union_areas

    return ious, union_areas


@torch.no_grad()
def post_process(preds_class: torch.Tensor, # (B, sum(H_p[i]*W_p[i]*anchor)), classes)
                 preds_box: torch.Tensor, # (B, sum(H_p[i]*W_p[i]*anchor)), 4)
                 anchors: torch.Tensor, # (levels=5, H_p[i]*W_p[i]*9, 4)
                 targets: dict,
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.5):
    """
    + アンカーボックスと予測誤差の統合
    + 余分な検出矩形の除去
    :param preds_class: 検出矩形のクラス (B, anchors, classes)
    :param preds_box: 検出矩形のアンカーボックスからの誤差 (B, anchors, 4)
    :param anchors: アンカーボックス (B, 4)
    :param targets: ラベル
    :param conf_threshold: 信頼度の閾値
    :param nms_threshold: NMSのIoU閾値
    :return:
    """
    batch_size = preds_class.shape[0]

    anchors_xywh = convert_to_xywh(anchors)

    # (B, 検出矩形数, 4[x,y,w,h])
    # 中心座標の予測をスケール不変にするため
    # 予測値をアンカーボックスの大きさでスケールする
    # ネットワークの出力=誤差と見なす.
    # 中心 -> (xr,yr) = (xa+xp*wa, ya+yp*ha)
    # 幅高 -> (wr,hr) = (wa*exp(wp), ha*exp(hp))
    # -> log(wr,hr) = (log(wp)+wa, log(hp)+ha) つまり誤差はlog空間の値
    preds_box[:, :, :2] = anchors_xywh[:, :2] + preds_box[:, :, :2] * anchors_xywh[:, 2:]  # (left,top)
    preds_box[:, :, 2:] = anchors_xywh[:, 2:] * preds_box[:, :, 2:].exp()  # (w,h)

    preds_box = convert_to_xyxy(preds_box)

    # 物体クラスの予測確率をシグモイド関数で計算
    # RetinaNetでは背景クラスは存在せず、
    # 背景を表す場合は、全ての物体クラスの予測確率が低くなるように実装されている
    preds_class = preds_class.sigmoid()

    # 画像ごとの処理
    scores = []
    labels = []
    boxes = []
    for (img_preds_class,
         img_preds_box,
         img_targets) in zip(preds_class, preds_box, targets):

        # 検出矩形が画像内に収まるように座標をクリップ
        img_preds_box[:, ::2] = img_preds_box[:, ::2].clamp(min=0, max=img_targets['size'][0])  # (xmin,ymin)
        img_preds_box[:, 1::2] = img_preds_box[:, 1::2].clamp(min=0, max=img_targets['size'][1])  # (xmax,ymax)

        # 検出矩形は入力画像の大きさに合わせたものになっているので、
        # 元々の画像に合わせて検出矩形をスケールする
        img_preds_box *= img_targets['orig_size'][0] / img_targets['size'][0]

        # 物体クラスのスコアとクラスIDを取得
        img_preds_score, img_preds_label = img_preds_class.max(dim=1) # (class_prob, class_index=id)

        # 信頼度がしきい値より高い検出矩形のみを残す
        keep = img_preds_score > conf_threshold
        img_preds_score = img_preds_score[keep]
        img_preds_label = img_preds_label[keep]
        img_preds_box = img_preds_box[keep]

        # クラス毎にNMSを適用
        keep_indices = batched_nms(img_preds_box,
                                   img_preds_score,
                                   img_preds_label,
                                   nms_threshold)

        scores.append(img_preds_score[keep_indices])
        labels.append(img_preds_label[keep_indices])
        boxes.append(img_preds_box[keep_indices])

    return scores, labels, boxes
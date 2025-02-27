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
def post_process(preds_class: torch.Tensor,
                 preds_box: torch.Tensor,
                 targets: dict,
                 include_bg: bool = False):
    """
    DETRで算出した矩形座標を原画像に戻す処理
    :param preds_class: 検出矩形クラス [B, Q, class + 1]
    :param preds_box: 検出矩形 [B, Q, 4]
    :param targets: ラベル
    :param include_bg: 分類結果に背景を含めるか否かのフラグ
    :return:
    """

    # クラス確率
    probs = preds_class.softmax(dim=2)

    if include_bg:
        scores, labels = probs.max(dim=2) # Value, Index
    else:
        scores, labels = probs[:, :, :-1].max(dim=2)

    # (cx, cy, w, h) -> (min_x, min_y, max_x, max_y)
    boxes = convert_to_xyxy(preds_box)

    # 矩形をミニバッチのサンプル毎の画像の大きさに合わせる
    img_sizes = torch.stack([target['orig_size'] for target in targets])
    boxes[:, :, ::2] *= img_sizes[:, 0].view(-1, 1, 1) # (B, Q, 1, 1) Width
    boxes[:, :, 1::2] *= img_sizes[:, 1].view(-1, 1, 1) # (B, Q, 1, 1) Height

    return scores, labels, boxes




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


# My libs
from utils import (
    convert_to_xywh,
    convert_to_xyxy,
    calc_iou,
)


"""損失"""

def calc_giou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Generalized IoU(GIoU)損失
    :param boxes1: 矩形集合 [矩形数, 4]
    :param boxes2: 矩形集合 [矩形数, 4]
    :return:
    """
    ious, union = calc_iou(boxes1, boxes2)

    # 2つの矩形を包含する最小の矩形の面積を計算
    left_top = torch.minimum(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])  # (N, 1, 2) * (M, 2) -> (N, M, 2)
    right_bottom = torch.maximum(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    areas = width_height.prod(dim=2)

    return ious - (areas - union) / areas



@torch.no_grad()
def _hungarian_match(preds_class: torch.Tensor,
                     preds_box: torch.Tensor,
                     targets: dict,
                     loss_weight_class: float = 1.0,
                     loss_weight_box_l1: float = 5.0,
                     loss_weight_box_giou: float = 2.0,
                     ):
    """
    コスト行列を最小化する組合せインデックスを取得するハンガリアンアルゴリズム
    :param preds_class: 検出矩形のクラス [B, Q, class + 1]
    :param preds_box: 検出矩形の位置と大きさ [B, Q, 4 (x, y, w, h)]
    :param targets: ラベル
    :param loss_weight_class: コストを計算する際の分類コストの重み
    :param loss_weight_box_l1: コストを計算する際の矩形のL1コストの重み
    :param loss_weight_box_giou: コストを計算する際の矩形のGIoUコストの重み
    :return:
    """
    batch_size, num_queries = preds_class.shape[:2] # (B,Q)

    # コスト計算を全てのサンプル一括で計算するために全てのサンプルの予測結果を一旦第0軸に並べる
    preds_class = preds_class.flatten(start_dim=0, end_dim=1).softmax(dim=1) # (B,Q,class+1) ->(BQ,class+1)
    preds_box = preds_box.flatten(start_dim=0, end_dim=1) # (B,Q,4) -> (BQ,4)

    # 全てのサンプルの正解ラベル(矩形)も一旦第0軸に並べる
    targets_class = torch.cat([target['classes'] for target in targets])

    # 正解矩形の値を正規化された画像上の座標に変換. (min_x, min_y, max_x, max_y) / (w, h, w, h)
    targets_box = torch.cat([ target['boxes'] / target['size'].repeat(2) for target in targets ])

    '''
    検出矩形と正解矩形の割り当てはサンプル(入力画像)毎におこなうため,
    サンプル毎のコスト行列が必要だが, コスト行列は、全サンプルの検出矩形と正解矩形を
    一括[全サンプルの検出矩形数, 全サンプルの正解矩形数]でまとめて作成する.
    最後に, サンプル毎のコスト行列に分割する. この方法だとGPUを最大限使用できる.
    '''

    # コスト[1]
    # 分類のコストは正解クラスの予測確率にマイナスをかけたもの
    # 正解クラスの予測確率が高ければ高いほどコストが小さくなる
    cost_class = -1 * preds_class[:, targets_class]  # [BQ(検出ラベル), M_j(正解ラベル)]

    # コスト[2]
    # 矩形回帰の1つ目のコストとなる予測結果と正解のL1誤差の計算
    # cdist関数は, (N,K)のテンソルと(M,K)のテンソルを与えられたとき,
    # (N,M)の各組合せでK個の要素を使ってL1誤差を計算し, そのL1誤差を保持した(N,M)テンソルを返す.
    cost_box_l1 = torch.cdist(preds_box, convert_to_xywh(targets_box), p=1) # (BQ, M)

    # コスト[3]
    # 矩形回帰の2つ目のコストとなる予測結果と正解のGIoU損失の計算
    cost_box_giou = -1 * calc_giou(convert_to_xyxy(preds_box), targets_box) # (BQ,M)

    cost = loss_weight_class * cost_class + \
           loss_weight_box_l1 * cost_box_l1 + \
           loss_weight_box_giou * cost_box_giou

    # コストを(BQ, M) -> (B, Q, M)に変更
    cost = cost.view(batch_size, num_queries, -1)

    # SciPyのlinear_sum_assignment関数を適用するためCPUへ転送
    cost = cost.to('cpu')

    # 各サンプルの正解矩形数を計算
    sizes = [len(target['classes']) for target in targets]

    '''
        各サンプル毎の[検出矩形, 正解矩形]の組のコスト行列に分解する
        '''
    indices = []
    # 第2軸を各サンプルの正解矩形数で分解し, バッチ軸でサンプルを
    # 指定することで, 各サンプルのコスト行列を取得
    for batch_id, c in enumerate(cost.split(sizes, dim=2)):
        c_batch = c[batch_id]  # コスト行列

        # ハンガリアンアルゴリズムによる予測結果と正解のマッチング
        # クエリのインデックスと正解のインデックスを得る
        pred_indices, target_indices = linear_sum_assignment(c_batch)

        indices.append(
            (torch.tensor(pred_indices, dtype=torch.int64),
             torch.tensor(target_indices, dtype=torch.int64))
        )

    return indices


def _get_pred_permutation_index(indices: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    バッチインデックスと検出矩形のインデックスのペアを作る並べ替え
    :param indices: ハンガリアンアルゴリズムにより得られたインデックス
    :return:
    """
    # マッチした予測結果のバッチインデックス(サンプルインデックス)を1つの軸に並べる
    batch_indices = torch.cat([
        torch.full_like(pred_indices, i) for i, (pred_indices, _) in enumerate(indices)
    ])

    # マッチした予測結果(検出矩形)のインデックスを1つの軸に並べる
    pred_indices = torch.cat([pred_indices for (pred_indices, _) in indices])

    return batch_indices, pred_indices


def _class_loss_func(preds: torch.Tensor,
                     targets: dict,
                     indices: List[Tuple[torch.Tensor, torch.Tensor]],
                     background_weight: float):
    """
    分類損失
    :param preds: 検出矩形のクラス [B, Q, class + 1]
    :param targets: ラベル
    :param indices: ハンガリアンアルゴリズムにより得られたインデックス
    :param background_weight: 背景クラスの交差エントロピー誤差の重み
    :return:
    """

    '''
    正解矩形を割り当てられた検出矩形には、正解クラスによる交差エントロピー誤差を計算.
    正解矩形を割り当てられなかった検出矩形に対しては、背景クラスによる交差エントロピー誤差を計算する
    '''
    pred_indices = _get_pred_permutation_index(indices)  # e.g. ([0,0,0,1,1,1], [2,1,0,0,2,1])

    # 物体クラス軸の最後の次元が背景クラス
    background_id = preds.shape[2] - 1

    # 正解ラベルとなるテンソルの作成
    # (B,Q)のテンソルを作成して背景IDを設定
    targets_class = preds.new_full(preds.shape[:2], background_id, dtype=torch.int64)

    # マッチした予測結果(矩形)の部分に正解ラベルの物体クラスIDを代入
    targets_class[pred_indices] = torch.cat([
        target['classes'][target_indices] for target, (_, target_indices) in zip(targets, indices)
    ])

    # 背景クラスの正解数が多く, クラス不均衡が生じるため背景クラスのコスト重みを下げる
    weights = preds.new_ones(preds.shape[2]) # (class + 1)
    weights[background_id] = background_weight

    # 交差エントロピー誤差 preds: (B, Q, class+1) -> (B, class+1, Q)
    loss = F.cross_entropy(preds.transpose(1,2), targets_class, weight=weights)

    return loss


def _box_loss_func(preds: torch.Tensor,
                   targets: dict,
                   indices: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    矩形損失, GIoU損失
    :param preds: 検出矩形の位置と大きさ [B, Q, 4(x,y,w,h)]
    :param targets: ラベル
    :param indices: ハンガリアンアルゴリズムにより得られたインデックス
    :return:
    """
    pred_indices = _get_pred_permutation_index(indices)

    # マッチした予測結果の抽出
    preds = preds[pred_indices]

    # マッチした正解を抽出
    targets_box = torch.cat([
        target['boxes'][target_indices] for target, (_, target_indices) in zip(targets, indices)
    ])

    # 0除算防止
    num_boxes = max(1, targets_box.shape[0])

    # L1誤差
    loss_l1 = F.l1_loss(preds, convert_to_xywh(targets_box), reduce='sum') / num_boxes

    # GIoUを計算
    gious = calc_giou(preds, targets_box)
    loss_giou = (1 - gious.diag()).sum() / num_boxes

    return loss_l1, loss_giou


def loss_func(preds_class: torch.Tensor,
              preds_box: torch.Tensor,
              targets: dict,
              loss_weight_class: float = 1.0,
              loss_weight_box_l1: float = 5.0,
              loss_weight_box_giou: float = 2.0,
              background_weight: float = 0.1):

    indices = _hungarian_match(preds_class,
                               preds_box,
                               targets,
                               loss_weight_class,
                               loss_weight_box_l1,
                               loss_weight_box_giou)

    loss_class = loss_weight_class * _class_loss_func(
        preds_class, targets, indices, background_weight
    )

    loss_box_l1, loss_box_giou = _box_loss_func(preds_box,
                                                targets,
                                                indices)

    loss_box_l1 = loss_weight_box_l1 * loss_box_l1
    loss_box_giou = loss_box_giou * loss_box_giou

    return loss_class, loss_box_l1, loss_box_giou
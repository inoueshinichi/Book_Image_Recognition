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


"""データ拡張"""

class RandomHorizontalFlip:

    def __init__(self, prob: float=0.5):
        self.prob = prob

    def __call__(self,
                 pil_img: Image,
                 target: dict,
                 ):
        if random.random() < self.prob:
            pil_img = TF.hflip(pil_img)

        # 正解矩形をx軸方向に反転
        # 制約式: max_x - min_x = width
        # min_x -> width - max_x
        # max_x -> width - min_x
        width = pil_img.size[0]
        # print("type(target)", type(target))
        # print('target', target)
        target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]

        return pil_img, target


class RandomSizeCrop:
    '''
    scale: 切り抜き前に対する切り抜き後の画像面積の下限と上限
    ratio: 切り抜き後の画像のアスペクト比の下限と上限
    '''
    def __init__(self,
                 scale: Sequence[float],
                 ratio: Sequence[float],
                 ):
        self.scale = scale
        self.ratio = ratio

    '''
    無作為に画像を切り抜く
    '''
    def __call__(self,
                 pil_img: Image,
                 target: dict,
                 ):
        width, height = pil_img.size

        # 切り抜く領域の左上の座標と幅および高さを取得
        # 切り抜く領域はscaleとratioの下限と上限に従う
        top, left, cropped_height, cropped_width = \
            T.RandomResizedCrop.get_params(pil_img,
                                          self.scale,
                                          self.ratio)
        # 左上の座標と幅および高さで指定した領域を切り抜き
        pil_img = TF.crop(pil_img, top, left, cropped_height, cropped_width)

        # 原点がx=left,y=topになるように矩形座標を平行移動
        # (min_x, min_y, max_x, max_y)
        target['boxes'][:, ::2] -= left # min_x, max_x
        target['boxes'][:, 1::2] -= top # min_y, max_y

        # 矩形の座標が切り抜き後に領域外に出る場合は座標をクリップする
        target['boxes'][:, ::2] = \
            target['boxes'][:, ::2].clamp(min=0) # min_x, max_x
        target['boxes'][:, 1::2] = \
            target['boxes'][:, 1::2].clamp(min=0) # min_y, max_y
        target['boxes'][:, ::2] = \
            target['boxes'][:, ::2].clamp(max=cropped_width) # min_x, max_x
        target['boxes'][:, 1::2] = \
            target['boxes'][:, 1::2].clamp(max=cropped_height) # min_y, max_y

        # 幅と高さが0より大きくなる矩形のみを保持(max_x > min_x & max_y > min_y)
        # (矩形数=画像内のインスタンス数, 1)
        keep = (target['boxes'][:, 2] > target['boxes'][:, 0]) & \
               (target['boxes'][:, 3] > target['boxes'][:, 1]) # マスク
        target['classes'] = target['classes'][keep]
        target['boxes'] = target['boxes'][keep]

        # 切り抜き後の画像の大きさを保持
        target['size'] = torch.tensor([cropped_width, cropped_height], dtype=torch.int64)

        return pil_img, target


class RandomResize:
    '''
    無作為に画像をアスペクト比を保持してリサイズするクラス
    min_sizes: 短辺の長さの候補、この中から無作為に長さを抽出
    max_size :  長辺の長さの最大値
    '''
    def __init__(self, min_sizes: Sequence[int], max_size: int):
        self.min_sizes = min_sizes
        self.max_side = max_size


    def _get_target_size(self,
                         min_side: int, # 短辺
                         max_side: int, # 長辺
                         target: int, # 目標となる短辺の長さ
                         ):
        # アスペクト比を保持して短辺をtargetに合わせる
        max_side = int(max_side * target / min_side)
        min_side = target

        # 長辺がmax_sideを超えている場合
        # アスペクト比を保持して長辺をmax_sizeに合わせる
        # このとき, 短辺は, (self.max_size / max_size)倍する
        # つまり, min_sideはtargetから更に短くなる
        if max_side > self.max_side:
            min_side = int(min_side * self.max_side / max_side)
            max_side = self.max_side

        return min_side, max_side

    def __call__(self,
                 pil_img: Image,
                 target: dict,
                 ):
        # 短編の長さを候補の中から無作為に抽出
        min_side = random.choice(self.min_sizes)

        width, height = pil_img.size

        # リサイズ後の大きさを取得
        # 幅と高さのどちらが短編であるか場合分け
        if width < height:
            resized_width, resized_height = self._get_target_size(
                min_side=width, max_side=height, target=min_side)
        else:
            resized_height, resized_width = self._get_target_size(
                min_side=height, max_side=width, target=min_side)

        # 指定した大きさに画像をリサイズ
        pil_img = TF.resize(pil_img, (resized_height, resized_width))

        # 正解矩形をリサイズ前後のスケールに合わせて変更
        ratio = resized_width / resized_height
        target['boxes'] *= ratio

        # リサイズ後の画像の大きさを保存
        target['size'] = torch.tensor(
            [resized_width, resized_height], dtype=torch.int64
        )

        return pil_img, target


class ToTensor:
    # PIL -> Tensor
    def __call__(self,
                 pil_img: Image,
                 target: dict,
                 ):
        ten_img = TF.to_tensor(pil_img)
        return ten_img, target


class Normalize:
    # mean(r,g,b)
    # std (r,g,b)
    def __init__(self,
                 mean: Sequence[float],
                 std: Sequence[float],
                 ):
        self.mean = mean
        self.std = std

    def __call__(self,
                 ten_img: torch.Tensor,
                 target: dict,
                 ):
        ten_img = TF.normalize(ten_img,
                          mean=self.mean,
                          std=self.std,
                          )
        return ten_img, target


class RandomSelect:
    # transform1: データ拡張1
    # transform2: データ拡張2
    # prob: データ拡張1が適用される確率
    def __init__(self,
                 transform1: Callable,
                 transform2: Callable,
                 prob: float=0.5,
                 ):
        self.transform1 = transform1
        self.transform2 = transform2
        self.prob = prob

    def __call__(self,
                 pil_img: Image,
                 target: dict,
                 ):
        if random.random() < self.prob:
            return self.transform1(pil_img, target)

        return self.transform2(pil_img, target)


class Compose:

    def __init__(self,
                 transforms: Sequence[Callable],
                 ):
        self.transforms = transforms

    def __call__(self,
                 pil_img: Image,
                 target: dict,
                 ):
        img: Union[Image, torch.Tensor] = pil_img # 初回だけPIL.Image. 以降はtorch.Tensor
        for transform in self.transforms:
            img, target = transform(img, target) # オリジナル アノテーション矩形も変形が必要

        return img, target


"""データセット用関数"""

def generate_subset(dataset: Dataset,
                        ratio: float,
                        random_seed: int = 0):
        '''
        データセットを分割するための2つの排反なインデックス集合を生成する関数
        dataset: 分割対象のデータセット
        ratio: 1つ目のセットに含めるデータ量の割合
        random_seed: シード値
        '''

        # サブセットの大きさ
        size = int(len(dataset) * ratio)

        indices = list(range(len(dataset)))

        # シャッフル
        random.seed(random_seed)
        random.shuffle(indices)

        indices1, indices2 = indices[:size], indices[size:]

        return indices1, indices2


def collate_func(batch: List[Tuple[torch.Tensor, dict]]):

    # ミニバッチの中で最も大きい画像サイズを取得
    max_height = 0
    max_width = 0
    for img, _ in batch:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    # 最も大きな画像サイズをバッチ数分用意
    imgs = batch[0][0].new_zeros((len(batch), 3, max_height, max_width))
    # 最も大きな画像サイズに対応するマスクを用意
    masks = batch[0][0].new_ones((len(batch), max_height, max_width), dtype=torch.bool)

    targets = []
    for i, (img, target) in enumerate(batch):
        height, width = img.shape[1:]
        imgs[i, :, :height, :width] = img
        masks[i, :height, :width] = False # 画像領域はFalse
        targets.append(target)

    return imgs, masks, targets


class CocoDetectionDataset(torchvision.datasets.CocoDetection):
    """
    物体検出用COCOデータセット読み込みクラス
    img_directory: 画像ファイルが保存されているディレクトリパス
    anno_file: アノテーションファイル
    transforms: データ拡張と整形を行うクラスインスタンス
    """
    def __init__(self,
                 img_directory: str,
                 anno_file: str,
                 transform: Optional[Callable] = None):
        super().__init__(img_directory, anno_file)

        self.transform = transform

        # カテゴリIDに欠番があるため、それを埋めてクラスIDを割り当て
        self.classes = []

        # 元々のクラスIDと新しく割り当てたクラスIDのマッピング
        self.coco_to_pred = {}
        self.pred_to_coco = {}
        for i, category_id in enumerate(sorted(self.coco.cats.keys())):
            self.classes.append(self.coco.cats[category_id]['name'])

            # category_id: 欠番のある1から始まるCocoラベル
            # i: 0からの再割り当てラベル
            self.coco_to_pred[category_id] = i
            self.pred_to_coco[i] = category_id

    def __getitem__(self, idx: int):
        # img: PIL.Image
        pil_img, target = super().__getitem__(idx)

        # 親クラスのコンストラクタでself.idsに画像IDが格納されているので取得
        img_id = self.ids[idx]

        # 物体の集合を1つの矩形でアノテーションしているものを除外
        # アノテーションjsonに`iscrowd`がないもの or iscrowd == 0 のもの
        target = [
            obj for obj in target if 'iscrowd' not in obj or obj['iscrowd'] == 0
        ]

        # 学習用の該当画像に写る物体クラスIDと矩形座標値を取得
        classes = torch.tensor(
            [self.coco_to_pred[obj['category_id']] for obj in target], dtype=torch.int64
        )
        boxes = torch.tensor(
            [obj['bbox'] for obj in target], dtype=torch.float32
        )

        # 正解矩形が0個のとき, boxes.shape == [0]となってしまうため
        # 第1軸に4を追加して軸数を第2軸の次元を揃える
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0,4))

        width, height = pil_img.size
        boxes[:, 2:] += boxes[:, :2] # (M, 4[min_x, min_y, max_x, max_y])

        # 矩形が画像領域内に収まるようにクリッピング
        boxes[:, ::2] = boxes[:, ::2].clamp(min=0, max=width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

        # 学習画像の情報や
        # 1枚の学習データに対する複数正解矩形に関するアノテーションデータ
        target = {
            'image_id': torch.tensor(img_id, dtype=torch.int64),
            'classes': classes,
            'boxes': boxes,
            'size': torch.tensor((width, height), dtype=torch.int64),
            'orig_size': torch.tensor((width, height), dtype=torch.int64),
            'orig_img': torch.tensor(np.asarray(pil_img))
        }

        # データ拡張
        if self.transform is not None:
            pil_img, target = self.transform(pil_img, target)

        return pil_img, target

    def to_coco_label(self, label: int):
        """
        モデルで予測されたクラスIDからCOCOのクラスIDに変換する関数
        label: 予測されたクラスID
        :param label:
        :return:
        """
        return self.pred_to_coco[label]
    

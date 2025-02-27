"""
Image Caption
「SHOW AND TELL」 (LSTM) with MS_COCO 2014
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
from collections import (
    deque, Counter
)
import pickle
import datetime
from pprint import pprint

import numpy as np
# from scipy.optimize import linear_sum_assignment # ハンガリアンアルゴリズム
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
from torch.nn.utils.rnn import pack_padded_sequence

# torchvision
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.datasets import CocoCaptions
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
from torchvision import models

# MS_COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


"""定数"""
WORD_TO_ID_SAVE_PATH: str = os.path.join(os.getcwd(), 'coco2014_val_word_to_id.pkl')
ID_TO_WORD_SAVE_PATH: str = os.path.join(os.getcwd(), 'coco2014_val_id_to_word.pkl')


"""ヘルパー関数"""

def tokenize_caption(caption: str,
                     word_to_id: Dict[str, int],
                     ):
    """
    文書(caption)を単語IDのリスト(tokens_id)に変換
    :param caption:
    :param word_to_id:
    :return:
    """
    tokens = caption.lower().split()
    tokens_temp = []
    # 単語についたピリオドとカンマを削除
    for token in tokens:
        if token == '.' or token == ',':
            continue

        token = token.rstrip('.')
        token = token.rstrip(',')

        tokens_temp.append(token)

    tokens = tokens_temp

    # 文章を単語IDリストに変換
    tokens_ext = ['<start>'] + tokens + ['<end>']
    tokens_id = []
    for k in tokens_ext:
        if k in word_to_id:
            tokens_id.append(word_to_id[k])
        else:
            tokens_id.append(word_to_id['<unk>'])

    return torch.Tensor(tokens_id)



"""データセット用関数"""

def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2


def collate_func(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    """
    ミニバッチ取得時の関数
    :param batch: サンプルした複数画像とラベル(caption)をまとめたもの
    :param word_to_id: 単語 -> ID 方向の辞書
    :return:
    """
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [tokenize_caption(random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並べ替え
    batch = zip(imgs, captions)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    # 各キャプションの長さ
    lengths = [cap.shape[0] for cap in captions]

    # 最大のキャプション長での(B, max_S)テンソルを作り, <null>で埋める(padding)
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)

    for i, cap in enumerate(captions):
        end = lengths[i] # <end>
        targets[i, :end] = cap[:end]

    return imgs, targets, lengths





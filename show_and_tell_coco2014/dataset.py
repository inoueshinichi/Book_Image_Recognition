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

# My libs
from utils import (
    IMGS_DIR_PATH,
    ANNO_FILE_PATH,
    WORD_TO_ID_SAVE_PATH,
    ID_TO_WORD_SAVE_PATH,
)


def make_dictionary():
    # 辞書の作成

    # キャプションの読み込み
    coco = COCO(ANNO_FILE_PATH)
    anns_keys = coco.anns.keys()

    # 単語-IDのマップを作成
    coco_token = []
    for key in anns_keys:
        caption = coco.anns[key]['caption']
        tokens = caption.lower().split()
        coco_token.extend(tokens)

    # debug
    # pprint(coco_token)
    vocab_size: int = len(coco_token)
    print(f"vocab_size", vocab_size)

    # ピリオド、カンマを削除
    table = str.maketrans({
        '.': '',
        ',': '',
    })
    for k in range(vocab_size):
        coco_token[k] = coco_token[k].translate(table)

    # 単語ヒストグラム
    freq = Counter(coco_token)
    # pprint(freq)

    # 3回以上出現する単語のみに絞る
    filtered_vocab = [token for token, count in freq.items() if count >= 3]
    sorted(filtered_vocab)
    print(filtered_vocab)

    # 特殊トークンの追加
    filtered_vocab.append('<start>')
    filtered_vocab.append('<end>')
    filtered_vocab.append('<unk>')
    filtered_vocab.append('<null>')

    # 単語とIDのマップを作成
    word_to_id = { token: i for i, token in enumerate(filtered_vocab) }
    id_to_word = { i: token for i, token in enumerate(filtered_vocab) }

    # output
    with open(WORD_TO_ID_SAVE_PATH, 'wb') as f:
        pickle.dump(word_to_id, f)
    with open(ID_TO_WORD_SAVE_PATH, 'wb') as f:
        pickle.dump(id_to_word, f)

    print("単語数: ", len(word_to_id))



if __name__ == "__main__":
    make_dictionary()
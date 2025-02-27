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


"""損失"""

def loss_func(x, y, word_to_id):
    return F.cross_entropy(x, y, ignore_index=word_to_id.get('<null>', None))


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
import datetime

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

from dataset import (
    generate_subset,
    SubsetRandomSampler,
    CocoDetectionDataset,
    Compose,
    RandomHorizontalFlip,
    RandomResize,
    RandomSelect,
    RandomSizeCrop,
    ToTensor,
    Normalize,
    collate_func,
)

from arch import RetinaNet

from loss import loss_func


IMGS_DIR_PATH: str = r"F:\MS_COCO\2014\naive_dataset\val2014"
ANNO_FILE_PATH: str = r"F:\MS_COCO\2014\tasks\instances_val2014_small.json"
MODEL_SAVE_PATH: str = os.path.join(os.getcwd(), "model")

class ConfigTrainEval:
    def __init__(self):
        self.img_directory = IMGS_DIR_PATH
        self.anno_file = ANNO_FILE_PATH
        self.save_dir = MODEL_SAVE_PATH
        self.val_ratio = 0.2
        self.num_epochs = 1
        self.lr_drop = 45
        self.val_interval = 5
        self.lr = 1e-5
        self.clip = 0.1 # 勾配クリップ上限
        self.moving_avg = 100 # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 8
        self.num_workers = 2
        self.device = 'cuda'


def evaluate(data_loader: DataLoader,
             model: nn.Module,
             loss_func: Callable,
             val_loss_best: float,
             val_loss_file: str,
             epoch: int,
             config,
             conf_threshold: float = 0.05,
             nms_threshold: float = 0.5):
    """

    :param data_loader: 評価に使うデータを読み込むデータローダ
    :param model: 評価対象のモデル
    :param loss_func: 目的関数
    :return:
    """
    model.eval()

    losses_class = []
    losses_box = []
    losses = []
    preds = []
    img_ids = []
    for imgs, targets in tqdm(data_loader, desc='[Validation]'):
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            targets = [{
                k: v.to(model.get_device())
                    for k, v in target.items()
            } for target in targets]

            preds_class, preds_box, anchors = model(imgs)

            loss_class, loss_box = loss_func(preds_class,
                                             preds_box,
                                             anchors,
                                             targets)
            loss = loss_class + loss_box

            losses_class.append(loss_class)
            losses_box.append(loss_box)
            losses.append(loss)

            """評価時は学習時と異なり異なり、
            損失を計算した後にpost_process関数を使って後処理をする"""
            scores, labels, boxes = post_process(preds_class,
                                                 preds_box,
                                                 anchors,
                                                 targets,
                                                 conf_threshold=conf_threshold,
                                                 nms_threshold=nms_threshold)

            # 入力画像毎の処理
            for (img_scores,
                 img_labels,
                 img_boxes,
                 img_targets) in zip(scores, labels, boxes, targets):

                img_ids.append(img_targets['image_id'].item())

                # 評価のためにCOCOの元々の矩形表現である
                # cx, cy, width, height に変換
                img_boxes[:, 2:] -= img_boxes[:, :2]

                # 検出矩形毎の処理
                for (score,
                     label,
                     box) in zip(img_scores, img_labels, img_boxes):

                    # COCO評価用のデータの保存
                    preds.append({
                        'image_id': img_targets['image_id'].item(),
                        'category_id': data_loader.dataset.to_coco_label(label.item()),
                        'score': score.item(),
                        'bbox': box.to('cpu').numpy().tolist()
                    })

    loss_class = torch.stack(losses_class).mean().item()
    loss_box = torch.stack(losses_box).mean().item()
    loss = torch.stack(losses).mean().item()
    print(f'Validation loss = {loss:.3f},'
          f'class loss = {loss_class:.3f}, '
          f'box loss = {loss_box:.3f} ')

    with open(val_loss_file, 'a') as f:
        print(f"{epoch}, {loss}", file=f)

    # Save model
    torch.save(model.state_dict(), os.path.join(config.save_dir, "retinanet.pth"))

    # より良い検証結果が得られた場合、モデルを保存
    if loss < val_loss_best:
        val_loss_best = loss

        torch.save(model.state_dict(), os.path.join(config.save_dir, f"retinanet_E{epoch}_best.pth"))

    if len(preds) == 0:
        print('Nothing detected, skip evaluation.')
        return

    # COCOevalクラスを使って評価するには検出結果を
    # jsonファイルに出力する必要があるため、jsonファイルに一時保存
    with open('tmp.json', 'w') as f:
        json.dump(preds, f)

    # 一時保存した検出結果をCOCOクラスを使って読み込み
    coco_results = data_loader.dataset.coco.loadRes('tmp.json')

    # COCOevalクラスを使って評価
    coco_eval = COCOeval(data_loader.dataset.coco, coco_results, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def learn():
    config = ConfigTrainEval()

    # データ拡張・整形クラスの設定
    min_sizes = (480, 512, 544, 576, 608)

    trn_transforms = Compose((
        RandomHorizontalFlip(),
        RandomSelect(
            RandomResize(min_sizes, max_size=1024),
            Compose((
                RandomSizeCrop(scale=(0.8, 1.0), ratio=(0.75, 1.333)),
                RandomResize(min_sizes, max_size=1024),
            ))
        ),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ))

    tst_transforms = Compose((
        # テストは短辺最大で実行
        RandomResize((min_sizes[-1],), max_size=1333),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ))

    trn_ds = CocoDetectionDataset(
        img_directory=config.img_directory,
        anno_file=config.anno_file,
        transform=trn_transforms,
    )

    val_ds = CocoDetectionDataset(
        img_directory=config.img_directory,
        anno_file=config.anno_file,
        transform=tst_transforms,
    )

    # Subset Sampler
    val_set, trn_set = generate_subset(trn_ds, config.val_ratio)

    print(f'学習セットのサンプル数: {len(trn_set)}')
    print(f'検証セットのサンプル数: {len(val_set)}')

    # 学習時にランダムにサンプルするためのサンプラー
    trn_sampler = SubsetRandomSampler(trn_set)

    # Dataloader
    trn_dl = DataLoader(
        trn_ds, batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=trn_sampler,
        collate_fn=collate_func
    )
    val_dl = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_set,
        collate_fn=collate_func,
    )

    # RetinaNetの生成
    model = RetinaNet(len(trn_ds.classes))
    # ResNet18をImageNetの学習済みモデルで初期化
    # 最後の全結合層がないなどのモデルの改変を許容するため、strict=False
    model.backbone.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
        strict=False)

    # モデルを指定デバイスに転送
    model.to(config.device)

    # Loss
    loss_func_lambda = lambda preds_class, preds_box, anchors, targets : \
        loss_func(preds_class, preds_box, anchors, targets)

    # Optimizerの生成
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # 指定したエポックで学習率を1/10に減衰するスケジューラを生成
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[config.lr_drop], gamma=0.1)

    # Monitoring
    now = datetime.datetime.now()
    trn_loss_file = '{}/detr_trn_loss_{}.csv'.format(
        config.save_dir, now.strftime('%Y%m%d_%H%M%S'))
    val_loss_file = '{}/detr_val_loss_{}.csv'.format(
        config.save_dir, now.strftime('%Y%m%d_%H%M%S'))

    val_loss_best = float('inf')
    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(trn_dl) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 移動平均計算用
            losses_class = deque()
            losses_box = deque()
            losses = deque()

            # 入力画像毎の処理
            for imgs, targets in pbar:
                imgs = imgs.to(model.get_device())
                targets = [{
                    k: v.to(model.get_device()) for k, v in target.items()
                } for target in targets]

                optimizer.zero_grad()

                preds_class, preds_box, anchors = model(imgs)

                loss_class, loss_box = loss_func_lambda(preds_class,
                                                        preds_box,
                                                        anchors,
                                                        targets)

                loss = loss_class + loss_box

                loss.backward()

                # 勾配全体のL2ノルムが上限を超えるとき上限値でクリップ
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip)

                optimizer.step()

                losses_class.append(loss_class.item())
                losses_box.append(loss_box.item())
                losses.append(loss.item())

                if len(losses) > config.moving_avg:
                    losses_class.popleft()
                    losses_box.popleft()
                    losses.popleft()

                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'loss_class': torch.Tensor(
                        losses_class).mean().item(),
                    'loss_box': torch.Tensor(
                        losses_box).mean().item()})

                with open(trn_loss_file, 'a') as f:
                    print(f"{epoch}, {loss.item()}", file=f)

        pass # training

        # スケジューラでエポック数をカウント
        scheduler.step()

        # 検証
        if epoch % config.val_interval == 0:
            evaluate(val_dl, model, loss_func,
                     val_loss_best, val_loss_file,
                     epoch, config)


if __name__ == "__main__":
    learn()
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
    Compose,
    RandomHorizontalFlip,
    RandomResize,
    RandomSelect,
    RandomSizeCrop,
    ToTensor,
    Normalize,
    CocoDetectionDataset,
    collate_func,
    generate_subset,
    SubsetRandomSampler,
)

from loss import loss_func

from arch import DETR


"""学習と検証"""
IMGS_DIR_PATH: str = r"F:\MS_COCO\2014\naive_dataset\val2014"
ANNO_FILE_PATH: str = r"F:\MS_COCO\instances_val2014_small.json"
MODEL_SAVE_PATH: str = os.path.join(os.getcwd(), "model")


class ConfigTrainEval:
    '''
    ハイパーパラメータとオプションの設定
    '''
    def __init__(self):
        self.img_directory = IMGS_DIR_PATH                 # 画像があるディレクトリ
        self.anno_file = ANNO_FILE_PATH                    # アノテーションファイルのパス
        self.save_dir = MODEL_SAVE_PATH                    # パラメータを保存するパス
        self.val_ratio = 0.2                               # 検証に使う学習セット内のデータの割合
        self.num_epochs = 1                                # 学習エポック数
        self.lr_drop = 90                                  # 学習率を減衰させるエポック
        self.val_interval = 5                              # 検証を行うエポック間隔
        self.lr = 1e-4                                     # 学習率
        self.lr_backbone = 1e-5                            # バックボーンネットワークの学習率
        self.weight_decay = 1e-4                           # 荷重減衰
        self.clip = 0.1                                    # 勾配のクリップ上限
        self.num_queries = 100                             # 物体クエリ埋め込みのクエリベクトル数
        self.dim_hidden = 256                              # Transformer内の特徴量次元
        self.num_heads = 8                                 # マルチヘッドアテンションのヘッド数
        self.num_encoder_layers = 6                        # Transformerエンコーダの層数
        self.num_decoder_layers = 6                        # Transformerデコーダの層数
        self.dim_feedforward = 2048                        # Transformer内のFNNの中間特徴量次元
        self.dropout = 0.1                                 # Transformer内のドロップアウト率
        self.loss_weight_class = 1                         # 分類損失の重み
        self.loss_weight_box_l1 = 5                        # 矩形回帰のL1誤差の重み
        self.loss_weight_box_giou = 2                      # 矩形回帰のGIoU損失の重み
        self.background_weight = 0.1                       # 背景クラスの重み
        self.moving_avg = 100                              # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 8                                # バッチサイズ
        self.num_workers = 2                               # データローダに使うCPUプロセスの数
        self.device = 'cuda'                               # 学習に使うデバイス


def test_show_image():
    file_path: str = os.path.join(IMGS_DIR_PATH, "COCO_val2014_000000441814.jpg")
    img = Image.open(file_path)
    img.show()


def test_transform_image():
    # データ拡張・整形クラスの設定
    min_sizes = (480, 512, 544, 576, 608)

    transforms = Compose((
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


    print("Naive Dataset")

    naive_trn_coco_ds = CocoDetectionDataset(
        img_directory=IMGS_DIR_PATH,
        anno_file=ANNO_FILE_PATH,
    )

    naive_record = naive_trn_coco_ds[0]
    naive_target = naive_record[1]
    print(naive_target['orig_img'].size())


    print("Transformed Dataset")

    transform_trn_coco_ds = CocoDetectionDataset(
        img_directory=IMGS_DIR_PATH,
        anno_file=ANNO_FILE_PATH,
        transform=transforms,
    )

    transform_record = transform_trn_coco_ds[0]
    transform_target = transform_record[1]
    print(transform_target['orig_img'].size())


def evaluate(data_loader,
             model,
             loss_func,
             val_loss_best,
             val_loss_file,
             epoch,
             config,
             ):
    """
    検証: DETR
    :param data_loader: 評価に使うデータを読み込むデータローダ
    :param model: 評価対象のモデル
    :param loss_func: 目的関数
    :param config: 設定パラメータ
    :return:
    """
    model.eval()

    losses_class = []
    losses_box_l1 = []
    losses_box_giou = []
    losses_aux = []
    losses = []
    preds = []
    img_ids = []
    for imgs, masks, targets in tqdm(data_loader, desc='[Validation]'):
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            masks = masks.to(model.get_device())
            targets = [
                { k: v.to(model.get_device()) for k, v in target.items() }
                for target in targets
            ]

            preds_class, preds_box = model(imgs, masks) # (Decoders,B,Q,class+1), (Decoders,B,Q,4)

            num_decoder_layers = preds_class.shape[0]

            loss_aux = 0
            for layer_index in range(num_decoder_layers - 1):
                loss_aux += sum(loss_func(preds_class[layer_index],
                                          preds_box[layer_index],
                                          targets))
                loss_class, loss_box_l1, loss_box_giou = loss_func(
                    preds_class[-1], preds_box[-1], targets
                ) # 分類損失, 矩形損失(L1,GIoU)は, Transformerの最終層の出力に対して計算する

                loss = loss_class + loss_box_l1 + loss_box_giou + loss_aux

                losses_class.append(loss_class)
                losses_box_l1.append(loss_box_l1)
                losses_box_giou.append(loss_box_giou)
                losses_aux.append(loss_aux)
                losses.append(loss)

                # 後処理により最終的な検出矩形を取得
                scores, labels, boxes = post_process(preds_class[-1],
                                                     preds_box[-1],
                                                     targets)

                # 原画像サンプル毎に分解
                for img_scores, img_labels, img_boxes, img_targets in zip(
                    scores, labels, boxes, targets):
                    img_ids.append(img_targets['image_id'].item())

                    # 評価のためにCOCOの元々の矩形表現である
                    # min_x, min_y, width, height に変換
                    img_boxes[:, 2:] -= img_boxes[:, 2:] # width = max_x - min_x

                    # 原画像に写る物体毎に分解
                    for score, label, box in zip(img_scores, img_labels, img_boxes):
                        # COCO評価用のデータの保存
                        preds.append({
                            'image_id': img_targets['image_id'].item(),
                            'category_id': data_loader.dataset.to_coco_label(label.item()),
                            'score': score.item(),
                            'bbox': box.to('cpu').numpy().tolist()
                        })

    pass # for

    loss_class = torch.stack(losses_class).mean().item()
    loss_box_l1 = torch.stack(losses_box_l1).mean().item()
    loss_box_giou = torch.stack(losses_box_giou).mean().item()
    loss_aux = torch.stack(losses_aux).mean().item()
    loss = torch.stack(losses).mean().item()
    print(f'Validation loss = {loss:.3f}, '
          f'class loss = {loss_class:.3f}, '
          f'box l1 loss = {loss_box_l1:.3f}, '
          f'box giou loss = {loss_box_giou:.3f}, '
          f'aux loss = {loss_aux:.3f}')

    with open(val_loss_file, 'a') as f:
        print(f"{epoch}, {loss}", file=f)

    # Save model
    torch.save(model.state_dict(), os.path.join(config.save_dir, "detr.pth"))

    # より良い検証結果が得られた場合、モデルを保存
    if loss < val_loss_best:
        val_loss_best = loss

        torch.save(model.state_dict(), os.path.join(config.save_dir, f"detr_E{epoch}_best.pth"))

    if len(preds) == 0:
        print('Nothing detected, skip evaluation.')

        return

    # pycocotoolsを使って評価するには検出結果をjsonファイルに出力する
    # 必要があるため、jsonファイルに一時保存
    with open('tmp.json', 'w') as f:
        json.dump(preds, f)

    # 一時保存した検出結果をpycocotoolsを使って読み込み
    coco_results = data_loader.dataset.coco.loadRes('tmp.json')

    # pycocotoolsを使って評価
    coco_eval = COCOeval(
        data_loader.dataset.coco, coco_results, 'bbox')
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

    # DETR(ResNet18 backbone)モデルの生成
    model = DETR(config.num_queries, config.dim_hidden,
                 config.num_heads, config.num_encoder_layers,
                 config.num_decoder_layers, config.dim_feedforward,
                 config.dropout, len(trn_ds.classes))
    model.backbone.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
        strict=False)

    # モデルを指定デバイスに転送
    model.to(config.device)

    # Loss (目的関数に予めハイパーパラメータをセット)
    loss_func_lambda = lambda preds_class, preds_box, targets: \
        loss_func(preds_class, preds_box, targets,
                  config.loss_weight_class,
                  config.loss_weight_box_l1,
                  config.loss_weight_box_giou,
                  config.background_weight)

    # Optimizerの生成, バックボーンとそうでないモジュールとの
    # パラメータで異なる学習率を適用
    params_backbone = []
    params_others = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if 'backbone' in name:
                params_backbone.append(parameter)
            else:
                params_others.append(parameter)
    param_groups = [
        {'params': params_backbone, 'lr': config.lr_backbone},
        {'params': params_others, 'lr': config.lr},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[config.lr_drop],
                                               gamma=0.1)

    # Monitoring
    now = datetime.datetime.now()
    trn_loss_file = '{}/show_and_tell_trn_loss_{}.csv'.format(
        config.save_dir, now.strftime('%Y%m%d_%H%M%S'))
    val_loss_file = '{}/show_and_tell_val_loss_{}.csv'.format(
        config.save_dir, now.strftime('%Y%m%d_%H%M%S'))

    # Epochループ
    val_loss_best = float('inf')
    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(trn_dl) as pbar:
            pbar.set_description(f"[Epoch: {epoch+1}]")

            # 移動平均計算用
            losses_class = deque()
            losses_box_l1 = deque()
            losses_box_giou = deque()
            losses_aux = deque()
            losses = deque()

            # learning
            for imgs, masks, targets in pbar:
                imgs = imgs.to(model.get_device())
                masks = masks.to(model.get_device())
                targets = [{
                    k: v.to(model.get_device())
                    for k, v in target.items()
                } for target in targets]

                # Zero
                optimizer.zero_grad()

                # Forward
                preds_class, preds_box = model(imgs, masks) # (Decoders,B,Q,class+1), (Decoders,B,Q,4)

                # Loss
                # 補助損失を計算
                loss_aux = 0
                for layer_index in range(config.num_decoder_layers - 1):
                    loss_aux += sum(
                        loss_func_lambda(preds_class[layer_index],
                                         preds_box[layer_index],
                                         targets)
                    )

                loss_class, loss_box_l1, loss_box_giou = \
                    loss_func_lambda(preds_class[-1], preds_box[-1], targets)

                loss = loss_aux + loss_class + loss_box_l1 + loss_box_giou

                # Backward
                loss.backward()

                # Clipping grad
                torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)

                # Update grad
                optimizer.step()

                losses_class.append(loss_class.item())
                losses_box_l1.append(loss_box_l1.item())
                losses_box_giou.append(loss_box_giou.item())
                losses_aux.append(loss_aux.item())
                losses.append(loss.item())

                if len(losses) > config.moving_avg:
                    losses_class.popleft()
                    losses_box_l1.popleft()
                    losses_box_giou.popleft()
                    losses_aux.popleft()
                    losses.popleft()

                pbar.set_postfix(
                    {'loss': torch.Tensor(losses).mean().item(),
                     'loss_class': torch.Tensor(
                         losses_class).mean().item(),
                     'loss_box_l1': torch.Tensor(
                         losses_box_l1).mean().item(),
                     'loss_box_giou': torch.Tensor(
                         losses_box_giou).mean().item(),
                     'loss_aux': torch.Tensor(
                         losses_aux).mean().item()})

                with open(trn_loss_file, 'a') as f:
                    print(f"{epoch}, {loss.item()}", file=f)

        pass # training

        # Validation
        if epoch % config.val_interval == 0:
            evaluate(val_dl, model, loss_func,
                     val_loss_best, val_loss_file,
                     epoch, config)

        # Vary learning rate
        scheduler.step()


if __name__ == "__main__":
    learn()
    # test_show_image()
    # test_transform_image()
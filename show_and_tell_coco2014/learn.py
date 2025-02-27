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


#############################################

from arch import (
    CNNEncoder,
    LSTMDecoder,    
)

from utils import (
    WORD_TO_ID_SAVE_PATH,
    ID_TO_WORD_SAVE_PATH,
    tokenize_caption,
    generate_subset,
    collate_func,
)

from loss import loss_func


"""学習と検証"""

IMGS_DIR_PATH: str = r"F:\MS_COCO\2014\naive_dataset\val2014"
ANNO_FILE_PATH: str = r"F:\MS_COCO\2014\naive_dataset\annotations_trainval2014\annotations\captions_val2014.json"
MODEL_SAVE_PATH: str = os.path.join(os.getcwd(), "model")

class ConfigTrain:
    def __init__(self):
        self.img_directory = IMGS_DIR_PATH
        self.anno_file = ANNO_FILE_PATH
        self.word_to_id_file = WORD_TO_ID_SAVE_PATH
        self.save_dir = MODEL_SAVE_PATH
        self.val_ratio = 0.3
        self.num_workers = 4
        self.device = 'cuda'
        self.moving_avg = 100
        self.dim_embedding = 300 # 埋め込み層の次元
        self.dim_hidden = 128 # LSTM隠れ層の次元
        self.num_layers = 2  # LSTM階層の数
        self.val_interval = 5  # 検証を行うエポック間隔
        self.lr = 0.001
        self.dropout = 0.3
        self.batch_size = 30
        self.num_epochs = 1 # 10 # 100
        self.lr_drop = [20] # 学習率を減衰させるエポック


def evaluate(data_loader,
             encoder,
             decoder,
             loss_func,
             val_loss_best,
             val_loss_file,
             epoch,
             config,
             ):

    with tqdm(data_loader) as pbar:
        pbar.set_description(f'[検証]')

        encoder.eval()
        decoder.eval()

        val_losses = []
        for imgs, captions, lengths in pbar:
            imgs = imgs.to(encoder.device)
            captions = captions.to(encoder.device)

            # Encoder/Decoder Model
            features = encoder(imgs)
            outputs = decoder(features, captions, lengths)

            # Loss
            targets = pack_padded_sequence(captions,
                                           lengths,
                                           batch_first=True)[0]
            val_loss = loss_func(outputs, targets)
            val_losses.append(val_loss.item())

            # Print loss
            val_loss = np.mean(val_losses)
            print(f"Validation loss: {val_loss}")

            # Writing loss to file
            with open(val_loss_file, 'a') as f:
                print(f"{epoch}, {val_loss.item()}", file=f)

            # Save models
            torch.save(encoder.state_dict(), os.path.join(config.save_dir, "show_and_tell_encoder.pth"))
            torch.save(decoder.state_dict(), os.path.join(config.save_dir, "show_and_tell_decoder.pth"))

            # より良い検証結果が得られた場合、モデルを保存
            if val_loss < val_loss_best:
                val_loss_best = val_loss

                torch.save(encoder.state_dict(), os.path.join(config.save_dir, f"show_and_tell_encoder_E{epoch}_best.pth"))
                torch.save(decoder.state_dict(), os.path.join(config.save_dir, f"show_and_tell_decoder_E{epoch}_best.pth"))


def learn():
    config = ConfigTrain()

    # 単語→単語ID
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    vocab_size = len(word_to_id)

    os.makedirs(config.save_dir, exist_ok=True)

    transforms = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # ImageNet標準化
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # coco dataset
    trn_ds = CocoCaptions(root=config.img_directory,
                           annFile=config.anno_file,
                           transform=transforms)

    val_set, trn_set = generate_subset(trn_ds, config.val_ratio)

    # 学習時にランダムにサンプルするためのサンプラー
    trn_sampler = SubsetRandomSampler(trn_set)

    # DataLoader
    collate_func_lambda = lambda x: collate_func(x, word_to_id)
    trn_dl = DataLoader(
        trn_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=trn_sampler,
        collate_fn=collate_func_lambda
    )
    val_dl = DataLoader(
        trn_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_set,
        collate_fn=collate_func_lambda
    )

    # Model
    encoder = CNNEncoder(config.dim_embedding)
    decoder = LSTMDecoder(config.dim_embedding,
                          config.dim_hidden,
                          vocab_size,
                          config.num_layers,
                          config.dropout)
    encoder.to(config.device)
    decoder.to(config.device)

    # Loss
    loss_func_lambda = lambda x, y: loss_func(x, y, word_to_id)

    # Optimizer
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config.lr_drop,
                                                     gamma=0.1)
    # Monitoring
    now = datetime.datetime.now()
    trn_loss_file = '{}/show_and_tell_trn_loss_{}.csv'.format(
        config.save_dir, now.strftime('%Y%m%d_%H%M%S'))
    val_loss_file = '{}/show_and_tell_val_loss_{}.csv'.format(
        config.save_dir, now.strftime('%Y%m%d_%H%M%S'))

    # training
    val_loss_best = float('inf')
    for epoch in range(config.num_epochs):
        with tqdm(trn_dl) as pbar:
            pbar.set_description(f"[Epoch] {epoch+1}")

            encoder.train()
            decoder.train()

            train_losses = deque()
            for imgs, captions, lengths in pbar:
                imgs = imgs.to(config.device)
                captions = captions.to(config.device)

                optimizer.zero()

                # Encoder/Decoder Model (Show and tell)
                features = encoder(imgs)
                outputs = decoder(features, captions, lengths)

                # Calculate loss
                targets = pack_padded_sequence(captions,
                                               lengths,
                                               batch_first=True)[0]
                loss = loss_func_lambda(outputs, targets)

                # Back Propagation
                loss.backward()

                # Update
                optimizer.step()

                # Writing loss to file
                train_losses.append(loss.item())
                if len(train_losses) > config.moving_avg:
                    train_losses.popleft()
                pbar.set_postfix({ 'loss': torch.Tensor(train_losses).mean().item() })
                with open(trn_loss_file, 'a') as f:
                    print(f"{epoch}, {loss.item()}", file=f)

        
        pass # training

        # validation
        if epoch % config.val_interval == 0:
            evaluate(val_dl, encoder, decoder,
                     loss_func_lambda, val_loss_best,
                     val_loss_file, epoch, config)
    
        # 学習率の調整
        scheduler.step()


if __name__ == "__main__":
    learn()

"""
Image Caption
「SHOW AND TELL」 (LSTM) with MS_COCO 2014
"""

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


"""NNモジュール"""

class CNNEncoder(nn.Module):
    """
    「Show and tell」 エンコーダ
    dim_embedding: 埋め込み次元
    """
    def __init__(self,
                 dim_embedding: int):
        super().__init__()

        # ImageNetで事前学習されたResNet152
        resnet = models.resnet152(weights="IMAGENET1K_V2")

        # 特徴抽出器として使うため全結合層を削除
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        self.linear = nn.Linear(resnet.fc.in_features,
                                dim_embedding) # (1,1,2048)

    def forward(self, imgs: torch.Tensor):
        """

        :param imgs: (B, C, H, W)
        :return:
        """

        # 特徴抽出 -> (B, 2048)
        # バックボーンは学習しない
        with torch.no_grad():
            features = self.backbone(imgs)
            features = features.flatten(1)

        # 全結合
        features = self.linear(features)

        return features


class LSTMDecoder(nn.Module):
    """
    Show and tell デコーダ
    dim_embedding: 埋め込み次元
    dim_hidden: 隠れ層次元
    vocab_size: 辞書サイズ
    num_layers: レイヤー数
    dropout: ドロップアウト率
    """
    def __init__(self,
                 dim_embedding: int,
                 dim_hidden: int,
                 vocab_size: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()

        # 単語埋め込み
        self.embed = nn.Embedding(vocab_size, dim_embedding)

        # LSTM
        self.lstm = nn.LSTM(input_size=dim_embedding,
                            hidden_size=dim_hidden,
                            num_layers=num_layers,
                            batch_first=True)

        # 全結合
        self.linear = nn.Linear(in_features=dim_hidden,
                                out_features=vocab_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                features: torch.Tensor,
                captions: torch.Tensor,
                lengths: list):
        """

        :param features: エンコーダ出力特徴 (B, D:埋め込み次元)
        :param captions: 画像キャプション (B, S:系列長)
        :param lengths: 系列長のリスト
        :return:
        """

        # 単語埋め込み: (B, S) -> (B, S, D)
        embeddings = self.embed(captions)

        # 画像埋め込みと単語埋め込みとを連結
        # features: (B, D) -> (B, 1, D)
        # 連結後: (B, S+1, D)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # パディングされたTensorを可変長系列に戻してパック
        # 系列長Sは, キャプション集合の中で最も長い固定値(<null>でpaddingされてる)
        # packed.data() -> (実際の系列長, D)
        packed = pack_padded_sequence(input=embeddings,
                                      lengths=lengths,
                                      batch_first=True
                                      )

        # LSTM
        hiddens, cell = self.lstm(packed)

        # ドロップアウト
        output = self.dropout(hiddens[0])

        # ロジット
        logits = self.linear(output)

        return logits

    @torch.no_grad()
    def sample(self,
               features: torch.Tensor,
               states: torch.Tensor=None,
               max_length: int=30):
        """
        サンプリングによる説明文出力(貪欲法)
        :param features: エンコーダ出力特徴 (B, D)
        :param states: LSTM隠れ状態
        :param max_length: キャプションの最大系列長
        :return:
        """
        inputs = features.unsqueeze(1) # (B, 1, D)
        word_idx_list = []

        # 最大系列長まで再帰的に単語をサンプリング予測
        for step_t in range(max_length):
            # LSTM隠れ状態を更新
            hiddens, states = self.lstm(inputs, states)

            # 単語予測
            outputs = self.linear(hiddens.squeeze(1)) # (B, V)
            probs = outputs.softmax(dim=1) # (B, V)
            preds = probs.argmax(dim=1) # (B,)
            word_idx_list.append(preds[0].item())

            # t+1の入力を作成
            inputs = self.embed(preds) # (B, D)
            inputs = inputs.unsqueeze(1) # (B, 1, D)

        return word_idx_list
    
    
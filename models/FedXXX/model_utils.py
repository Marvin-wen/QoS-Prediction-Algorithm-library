import torch
from torch import nn

from .resnet_utils import *


class Linear(nn.Module):
    def __init__(self, in_size, out_size, activation):
        super().__init__()
        self.fc_layer = nn.Sequential(nn.Linear(in_size, out_size),
                                      activation())

    def forward(self, x):
        x = self.fc_layer(x)
        return x


class Embedding(nn.Module):
    def __init__(self, type_, embedding_nums: list, embedding_dims: list):
        self.type = type_
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        assert self.type in ["stack", "cat"]
        super().__init__()
        self.embeddings = nn.ModuleList([
            *[
                nn.Embedding(num, dim)
                for num, dim in zip(embedding_nums, embedding_dims)
            ]
        ])

    def forward(self, indexes):

        if self.type == "stack":
            assert len(set(
                self.embedding_dims)) == 1, f"dims should be the same"

            x = sum([
                embedding(indexes[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ])
        elif self.type == "cat":
            x = torch.cat([
                embedding(indexes[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ],
                          dim=1)
        else:
            raise NotImplementedError
        return x


class SingleEncoder(nn.Module):
    def __init__(self,
                 type_,
                 embedding_nums: list,
                 embedding_dims: list,
                 in_size=128,
                 blocks_sizes=[64, 32, 16],
                 deepths=[2, 2, 2],
                 activation=nn.ReLU,
                 block=ResNetBasicBlock):
        super().__init__()
        # embedding

        self.embedding = Embedding(type_, embedding_nums, embedding_dims)

        # resnet encoder

        self.resnet_encoder = ResNetEncoder(in_size, blocks_sizes, deepths,
                                            activation, block)

    def forward(self, indexes: list):
        x = self.embedding(indexes)
        x = self.resnet_encoder(x)
        return x

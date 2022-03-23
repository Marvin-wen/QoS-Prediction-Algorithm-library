from collections import UserDict

import torch
from models.base import ModelBase
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_user, n_item, dim, layers=None, output_dim=1) -> None:
        """
        Args:
            n_user ([type]): 用户数量
            n_item ([type]): 物品数量
            dim ([type]): 特征空间的维度
            layers (list, optional): 多层感知机每层的维度. Defaults to [16,32,16,8].
            output_dim (int, optional): 最后输出的维度. Defaults to 1.
        """
        super(MLP, self).__init__()
        if layers is None:
            layers = [32, 16, 8]
        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items,
                                           embedding_dim=self.latent_dim)

        self.cf_layers = nn.ModuleList()

        # MLP的第一层
        # 输入是用户特征向量 + 项目特征向量，因为假定特征空间维度都是 latemt_dim，因此输入维度大小为 2 * latemt_dim
        self.cf_layers.append(nn.Linear(self.latent_dim * 2, layers[0]))

        for in_size, out_size in zip(layers, layers[1:]):
            self.cf_layers.append(nn.Linear(in_size, out_size))

        self.cf_output = nn.Linear(layers[-1], output_dim)

    def forward(self, user_idx, item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        for cf_layer in self.cf_layers:
            x = cf_layer(x)
            x = nn.ReLU()(x)
        x = self.cf_output(x)
        return x


class MLPModel(ModelBase):
    def __init__(self, loss_fn, n_user, n_item, dim, layers=None, output_dim=1, use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)
        if layers is None:
            layers = [32, 16, 8]
        self.model = MLP(n_user, n_item, dim, layers=layers, output_dim=output_dim)
        if use_gpu:
            self.model.to(self.device)
        self.name = __class__.__name__

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)

import torch
from torch import nn
from models.base.base import ModelBase
from .utils import ResNetEncoder,ResNetBasicBlock


class FedXXX(nn.Module):
    def __init__(self, user_params, item_params, linear_layers: list, output_dim=1, activation=nn.ReLU) -> None:
        super().__init__()

        # user
        self.user_encoder = SingleEncoder(
            **user_params)

        # item

        self.item_encoder = SingleEncoder(
            **item_params)

        # decoder
        self.fc_layers = nn.Sequential(
            *[Linear(in_size, out_size, activation) for in_size, out_size in zip(linear_layers, linear_layers[1:])]

        )
        # output
        self.output_layers = nn.Linear(linear_layers[-1], output_dim)

    def forward(self, user_idxes: list, item_idxes: list):
        user_feature = self.user_encoder(user_idxes)
        item_feature = self.item_encoder(item_idxes)
        x = torch.cat([user_feature, item_feature], dim=1)
        x = self.fc_layers(x)
        x = self.output_layers(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_size, out_size, activation):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_size, out_size),
            activation()
        )

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

            *[nn.Embedding(num, dim) for num, dim in zip(embedding_nums, embedding_dims)]

        ])

    def forward(self, indexes):
        if self.type == "stack":
            assert len(set(self.embedding_dims)) == 1, f"dims should be the same"
            x = sum([embedding(indexes[:,idx]) for idx,embedding in enumerate(self.embeddings)])
        elif self.type == "cat":
            x = torch.cat([embedding(indexes[:,idx]) for idx,embedding in enumerate(self.embeddings)], dim=1)
        else:
            raise NotImplementedError
        return x


class SingleEncoder(nn.Module):
    def __init__(self, type_, embedding_nums: list, embedding_dims: list, in_size=128, blocks_sizes=[64, 32, 16], deepths=[2, 2, 2], activation=nn.ReLU, block=ResNetBasicBlock):
        super().__init__()
        # embedding

        self.embedding = Embedding(type_, embedding_nums, embedding_dims)

        # resnet encoder

        self.resnet_encoder = ResNetEncoder(
            in_size, blocks_sizes, deepths, activation, block)

    def forward(self, indexes:list):
        x = self.embedding(indexes)
        x = self.resnet_encoder(x)
        return x


class FedXXXModel(ModelBase):
    def __init__(self, user_params, item_params, loss_fn, linear_layers: list, output_dim=1, activation=nn.ReLU,use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)

        self.model = FedXXX(user_params, item_params, linear_layers, output_dim, activation)

        if use_gpu:
            self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __str__(self) -> str:
        return str(self.model)

    def __repr__(self) -> str:
        return repr(self.model)

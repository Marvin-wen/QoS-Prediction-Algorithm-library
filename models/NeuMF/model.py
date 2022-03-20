import torch
from models.base import ModelBase
from torch import nn
import torch.nn.functional as F


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, layers=None, output_dim=1) -> None:
        super(NeuMF, self).__init__()

        # GMF网络的embedding层
        self.GMF_embedding_user = nn.Embedding(num_embeddings=num_users,
                                               embedding_dim=latent_dim)
        self.GMF_embedding_item = nn.Embedding(num_embeddings=num_items,
                                               embedding_dim=latent_dim)

        # MLP的embedding层
        self.MLP_embedding_user = nn.Embedding(num_embeddings=num_users,
                                               embedding_dim=latent_dim)
        self.MLP_embedding_item = nn.Embedding(num_embeddings=num_items,
                                               embedding_dim=latent_dim)

        # MLP网络
        self.MLP_layers = nn.ModuleList()
        # MLP第一层，输入是用户特征向量 + 项目特征向量，因为假定特征空间维度都是 latemt_dim，因此输入维度大小为 2 * latemt_dim
        self.MLP_layers.append(nn.Linear(latent_dim * 2, layers[0]))
        for in_size, out_size in zip(layers, layers[1:]):
            self.MLP_layers.append(nn.Linear(in_size, out_size))
        self.MLP_output = nn.Linear(layers[-1], latent_dim)

        # 合并模型
        self.linear = nn.Linear(2 * latent_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indexes, item_indexes):
        # GMF模型计算
        GMF_user_embedding = self.GMF_embedding_user(user_indexes)
        GMF_item_embedding = self.GMF_embedding_item(item_indexes)
        # 点积
        GMF_vec = torch.mul(GMF_user_embedding, GMF_item_embedding)

        # MLP模型计算
        MLP_user_embedding = self.MLP_embedding_user(user_indexes)
        MLP_item_embedding = self.MLP_embedding_item(item_indexes)
        # 隐向量堆叠
        x = torch.cat([MLP_user_embedding, MLP_item_embedding], dim=-1)
        # MLP网络
        for layer in self.MLP_layers:
            x = layer(x)
            x = F.relu(x)
        MLP_vec = self.MLP_output(x)

        # 合并模型
        vector = torch.cat([GMF_vec, MLP_vec], dim=-1)
        linear = self.linear(vector)
        output = self.sigmoid(linear)

        return output


class NeuMFModel(ModelBase):
    def __init__(self, loss_fn, num_users, num_items, latent_dim, layers=None, output_dim=1, use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)
        if layers is None:
            layers = [32, 16, 8]
        self.model = NeuMF(num_users, num_items, latent_dim, layers=layers, output_dim=output_dim)
        if use_gpu:
            self.model.to(self.device)
        self.name = __class__.__name__

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)

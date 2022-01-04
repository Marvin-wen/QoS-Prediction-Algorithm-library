import torch
from models.base import ModelBase
from torch import nn


class GMF(nn.Module):
    def __init__(self, n_user, n_item, dim=8, output_dim=1) -> None:
        super(GMF, self).__init__()

        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items,
                                           embedding_dim=self.latent_dim)

        self.fc_output = nn.Linear(self.latent_dim, output_dim)

    def forward(self, user_idx, item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)
        element_product = torch.mul(user_embedding, item_embedding)
        x = self.fc_output(element_product)
        return x


class GMFModel(ModelBase):
    def __init__(self,
                 loss_fn,
                 n_user,
                 n_item,
                 dim=8,
                 output_dim=1,
                 use_gpu=True):
        super().__init__(loss_fn, use_gpu)
        self.model = GMF(n_user, n_item, dim, output_dim)
        if use_gpu:
            self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)

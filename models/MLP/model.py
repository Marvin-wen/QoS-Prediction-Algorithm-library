from collections import UserDict
import torch
from torch import nn
from models.base import ModelBase

class MLP(nn.Module):
    def __init__(self,n_user,n_item,dim,layers=[32,16,8],output_dim=1) -> None:
        """

        Args:
            n_user ([type]): 用户数量
            n_item ([type]): 物品数量
            dim ([type]): 特征空间的维度
            layers (list, optional): 多层感知机的层数. Defaults to [16,32,16,8].
            output_dim (int, optional): 最后输出的维度. Defaults to 1.
        """
        super(MLP,self).__init__()
        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim


        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.latent_dim * 2,layers[0])) # MLP的第一层为latent vec的cat

        for in_size,out_size in zip(layers[:-1],layers[1:]):
            self.fc_layers.append(nn.Linear(in_size,out_size))
        
        self.fc_output = nn.Linear(layers[-1],output_dim)

    def forward(self,user_idx,item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)
        x = torch.cat([user_embedding,item_embedding],dim=-1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.ReLU()(x)
        x = self.fc_output(x)
        return x


class MLPModel(ModelBase):
    def __init__(self, loss_fn, n_user, n_item, dim, layers=[32, 16, 8], output_dim=1,use_gpu=True) -> None:
        self.model = MLP(n_user, n_item, dim, layers=layers, output_dim=output_dim)
        self.device = ("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        if use_gpu:
            self.model.to(self.device) 
        super().__init__(loss_fn)
    
    def parameters(self):
        return self.model.parameters()
    
    def __repr__(self) -> str:
        return str(self.model)




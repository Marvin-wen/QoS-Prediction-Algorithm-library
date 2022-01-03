from collections import UserDict
import torch
from torch import nn
from tqdm import tqdm
from .client import Clients
from .server import Server
from torch.optim.adam import Adam
from utils.mylogger import TNLog
from utils.evaluation import mae,mse,rmse
import numpy as np


class FedMLP(nn.Module):
    def __init__(self,n_user,n_item,dim,layers=[32,16,8],output_dim=1) -> None:
        """
        Args:
            n_user ([type]): 用户数量
            n_item ([type]): 物品数量
            dim ([type]): 特征空间的维度
            layers (list, optional): 多层感知机的层数. Defaults to [16,32,16,8].
            output_dim (int, optional): 最后输出的维度. Defaults to 1.
        """
        super(FedMLP,self).__init__()
        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.latent_dim * 2,layers[0])) # MLP的第一层为latent vec的cat

        for in_size,out_size in zip(layers,layers[1:]):
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


class FedMLPModel():
    def __init__(self, traid, loss_fn, n_user, n_item, dim, layers=[3], output_dim=1,use_gpu=True,optimizer=Adam) -> None:
        self.use_gpu = use_gpu
        self.name = __class__.__name__
        self.server = Server()
        self._model = FedMLP(n_user, n_item,
                             dim, layers, output_dim)
        self.clients = Clients(traid, self._model)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = TNLog(self.name)
        self.logger.initial_logger()


    
    def fit(self,epochs,lr,test_traid):
        for epoch in tqdm(range(epochs),desc="Training Epochs"):

            collector = []
            loss_list = []

            s_params = self.server.params if epoch != 0 else self._model.state_dict()

            for client_id,client in tqdm(self.clients,desc="Client training"):
                u_params,loss = client.fit(
                    s_params,self.loss_fn,self.optimizer,lr
                )
                collector.append(u_params)
                loss_list.append(loss)
                # print(client_id,loss)
                # import time
                # time.sleep(1)
            # print(np.average(loss_list))

            self.server.upgrade(collector)

            if (epoch+1) % 10 == 0:
                y_list,y_pred_list = self.predict(test_traid)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                self.logger.info(f"mae:{mae_},mse:{mse_},rmse:{rmse_}")

    def predict(self,test_loader):
        self.device = ("cuda" if (
            self.use_gpu and torch.cuda.is_available()) else "cpu")
        s_params = self.server.params
        self._model.load_state_dict(s_params)
        y_pred_list = []
        y_list = []
        self._model.eval()
        with torch.no_grad():
            for batch_id,batch in tqdm(enumerate(test_loader), desc="Model Predict"):
                user,item,rate = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_pred = self._model(user,item).squeeze()
                if len(y_pred.shape) == 0: # 64一batch导致变成了标量
                    y_pred = y_pred.unsqueeze(dim=0)
                y_pred_list.append(y_pred)
            y_pred_list = torch.cat(y_pred_list).cpu().numpy()
            y_list.append(rate)

        return y_list,y_pred_list

    def parameters(self):
        return self._model.parameters()
    
    def __repr__(self) -> str:
        return str(self._model)




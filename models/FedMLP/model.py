import random
from collections import UserDict

import numpy as np
import torch
from numpy.lib.function_base import select
from pandas.io.parsers import read_table
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse
from utils.mylogger import TNLog

from .client import Clients
from .server import Server


class FedMLP(nn.Module):
    def __init__(self,
                 n_user,
                 n_item,
                 dim,
                 layers=[32, 16, 8],
                 output_dim=1) -> None:
        """
        Args:
            n_user ([type]): 用户数量
            n_item ([type]): 物品数量
            dim ([type]): 特征空间的维度
            layers (list, optional): 多层感知机的层数. Defaults to [16,32,16,8].
            output_dim (int, optional): 最后输出的维度. Defaults to 1.
        """
        super(FedMLP, self).__init__()
        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items,
                                           embedding_dim=self.latent_dim)

        self.fc_layers = nn.ModuleList()
        # MLP的第一层为latent vec的cat
        self.fc_layers.append(nn.Linear(self.latent_dim * 2, layers[0]))

        for in_size, out_size in zip(layers, layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.fc_output = nn.Linear(layers[-1], output_dim)

    def forward(self, user_idx, item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.ReLU()(x)
        x = self.fc_output(x)
        return x


class FedMLPModel():
    def __init__(self,
                 traid,
                 loss_fn,
                 n_user,
                 n_item,
                 dim,
                 layers=[32, 16, 8],
                 output_dim=1,
                 use_gpu=True,
                 optimizer="adam") -> None:
        self.device = ("cuda" if
                       (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = __class__.__name__
        self._model = FedMLP(n_user, n_item, dim, layers, output_dim)

        self.server = Server()
        self.clients = Clients(traid, self._model, self.device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = TNLog(self.name)
        self.logger.initial_logger()

    def _check(self, iterator):
        assert abs(sum(iterator) - 1) <= 1e-4

    def update_selected_clients(self, sampled_client_indices, lr, s_params):
        """使用 client.fit 函数来训练被选择的client
        """
        collector = []
        client_loss = []
        selected_total_size = 0  # client数据集总数

        for uid in tqdm(sampled_client_indices, desc="Client training"):
            s_params, loss = self.clients[uid].fit(s_params, self.loss_fn,
                                                   self.optimizer, lr)
            collector.append(s_params)
            client_loss.append(loss)
            selected_total_size += self.clients[uid].n_item
        return collector, client_loss, selected_total_size

    def evaluate_selected_clients(self, sampled_client_indices):
        for uid in sampled_client_indices:
            self.clients[uid].evaluate()

    # todo how to add loss?
    def fit(self, epochs, lr, test_traid, fraction=0.1):
        for epoch in tqdm(range(epochs), desc="Training Epochs"):

            # 0. Get params from server
            s_params = self.server.params if epoch != 0 else self._model.state_dict(
            )

            # 1. Select some clients
            sampled_client_indices = self.clients.sample_clients(fraction)

            # 2. Selected clients train
            collector, loss_list, selected_total_size = self.update_selected_clients(
                sampled_client_indices, lr, s_params)

            # 3. Update params to Server
            mixing_coefficients = [
                self.clients[idx].n_item / selected_total_size
                for idx in sampled_client_indices
            ]
            self._check(mixing_coefficients)
            self.server.upgrade_wich_cefficients(collector,
                                                 mixing_coefficients)

            self.logger.info(
                f"[{epoch}/{epochs}] Loss:{sum(loss_list)/len(loss_list):>3.5f}"
            )

            print(list(self.clients[0].loss_list))

            if (epoch + 1) % 50 == 0:
                y_list, y_pred_list = self.predict(test_traid)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                self.logger.info(
                    f"Epoch:{epoch+1} mae:{mae_},mse:{mse_},rmse:{rmse_}")

    def predict(self, test_loader):
        s_params = self.server.params
        self._model.load_state_dict(s_params)
        y_pred_list = []
        y_list = []
        self._model.eval()
        with torch.no_grad():
            for batch_id, batch in tqdm(enumerate(test_loader),
                                        desc="Model Predict"):
                user, item, rate = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_pred = self._model(user, item).squeeze()
                y_real = rate.reshape(-1, 1)
                if len(y_pred.shape) == 0:  # 64一batch导致变成了标量
                    y_pred = y_pred.unsqueeze(dim=0)
                if len(y_real.shape) == 0:
                    y_real = y_real.unsqueeze(dim=0)
                y_pred_list.append(y_pred)
                y_list.append(y_real)
            y_pred_list = torch.cat(y_pred_list).cpu().numpy()
            y_list = torch.cat(y_list).cpu().numpy()

        return y_list, y_pred_list

    def parameters(self):
        return self._model.parameters()

    def __repr__(self) -> str:
        return str(self._model)

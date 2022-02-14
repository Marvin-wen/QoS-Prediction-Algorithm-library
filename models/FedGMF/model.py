import random
from collections import UserDict

import numpy as np
import torch
from models.base import FedModelBase, ModelBase
from numpy.lib.function_base import select
from pandas.io.parsers import read_table
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse
from utils.model_util import load_checkpoint, save_checkpoint
from utils.mylogger import TNLog

from .client import Clients
from .server import Server


class FedGMF(nn.Module):
    def __init__(self, n_user, n_item, dim=8, output_dim=1) -> None:
        super(FedGMF, self).__init__()

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


class FedGMFModel(FedModelBase):
    def __init__(self,
                 traid,
                 loss_fn,
                 n_user,
                 n_item,
                 dim,
                 output_dim=1,
                 use_gpu=True,
                 optimizer="adam") -> None:
        self.device = ("cuda" if
                       (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = __class__.__name__
        self._model = FedGMF(n_user, n_item, dim, output_dim)

        self.server = Server()
        self.clients = Clients(traid, self._model, self.device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = TNLog(self.name)
        self.logger.initial_logger()


    def evaluate_selected_clients(self, sampled_client_indices):
        for uid in sampled_client_indices:
            self.clients[uid].evaluate()

    # todo how to add loss?
    def fit(self, epochs, lr, test_traid, fraction=1):
        best_train_loss = None
        is_best = False
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
            if not best_train_loss:
                best_train_loss = sum(loss_list) / len(loss_list)
                is_best = True
            elif sum(loss_list) / len(loss_list) < best_train_loss:
                best_train_loss = sum(loss_list) / len(loss_list)
                is_best = True
            else:
                is_best = False

            ckpt = {
                "model": self.server.params,
                "epoch": epoch + 1,
                "best_loss": best_train_loss
            }
            save_checkpoint(ckpt, is_best, f"output/{self.name}",
                            f"loss_{best_train_loss:.4f}.ckpt")

            if (epoch + 1) % 20 == 0:
                y_list, y_pred_list = self.predict(test_traid)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                self.logger.info(
                    f"Epoch:{epoch+1} mae:{mae_},mse:{mse_},rmse:{rmse_}")

    def predict(self, test_loader, resume=False, path=None):
        if resume:
            ckpt = load_checkpoint(path)
            s_params = ckpt["model"]
            self._model.load_state_dict(s_params)
            self.logger.debug(
                f"Check point restored! => loss {ckpt['best_loss']:>3.5f} Epoch {ckpt['epoch']}"
            )
        else:
            s_params = self.server.params
            self._model.load_state_dict(s_params)

        self._model.load_state_dict(s_params)
        y_pred_list = []
        y_list = []
        self._model.to(self.device)
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

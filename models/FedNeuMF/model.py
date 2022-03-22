from pickletools import optimize
from xmlrpc.client import Server

import torch
import torch.nn.functional as F
from models.base.fedbase import FedModelBase
from models.FedGMF.client import Clients
from torch import nn
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse
from utils.model_util import load_checkpoint, save_checkpoint
from utils.mylogger import TNLog


class FedNeuMF(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 latent_dim,
                 layers=None,
                 output_dim=1) -> None:
        super(FedNeuMF, self).__init__()

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


class FedNeuMFModel(FedModelBase):
    def __init__(self,
                 triad,
                 loss_fn,
                 n_user,
                 n_item,
                 use_gpu=True,
                 optimizer="adam") -> None:
        super().__init__()
        self.device = ("cuda" if
                       (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = __class__.__name__
        self._model = FedNeuMF(loss_fn, n_user, n_item, use_gpu)
        self.server = Server()
        self.clients = Clients(triad, self._model, self.device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = TNLog(self.name)
        self.logger.initial_logger()

    def _check(self, iterator):
        assert abs(sum(iterator) - 1) <= 1e-4

    def fit(self, epochs, lr, test_triad, fraction=1):
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
                y_list, y_pred_list = self.predict(test_triad)
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

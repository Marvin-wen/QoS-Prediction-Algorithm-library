import collections
import torch
from torch import nn
import numpy as np
from torch.optim.adam import Adam
from models.base.base import ModelBase
from .utils import ResNetEncoder, ResNetBasicBlock
from .client import Clients
from .server import Server
from tqdm import tqdm
from utils.model_util import split_d_traid,save_checkpoint
from utils.evaluation import mae,mse,rmse


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

    def forward(self, user_idxes: list, item_idxes: list, need_feature=False):
        user_feature = self.user_encoder(user_idxes)
        item_feature = self.item_encoder(item_idxes)
        x = torch.cat([user_feature, item_feature], dim=1)
        x = self.fc_layers(x)
        x = self.output_layers(x)
        if need_feature:
            return x, user_feature, item_feature
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
            assert len(set(self.embedding_dims)
                       ) == 1, f"dims should be the same"

            x = sum([embedding(indexes[:, idx])
                    for idx, embedding in enumerate(self.embeddings)])
        elif self.type == "cat":
            x = torch.cat([embedding(indexes[:, idx])
                          for idx, embedding in enumerate(self.embeddings)], dim=1)
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

    def forward(self, indexes: list):
        x = self.embedding(indexes)
        x = self.resnet_encoder(x)
        return x

# 非联邦


class FedXXXModel(ModelBase):
    def __init__(self, user_params, item_params, loss_fn, linear_layers: list, output_dim=1, activation=nn.ReLU, use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)
        self.model = FedXXX(user_params, item_params,
                            linear_layers, output_dim, activation)
        if use_gpu:
            self.model.to(self.device)
        self.name = __class__.__name__

    def parameters(self):
        return self.model.parameters()

    def __str__(self) -> str:
        return str(self.model)

    def __repr__(self) -> str:
        return repr(self.model)


# 联邦
class FedXXXLaunch:
    """负责将client和server的交互串起来
    """

    def __init__(self, d_traid, user_params, item_params, linear_layers, loss_fn, output_dim=1, activation=nn.ReLU, optimizer=Adam) -> None:
        self.server = Server()
        self._model = FedXXX(user_params, item_params,
                             linear_layers, output_dim, activation)
        self.clients = Clients(d_traid, self._model)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, epochs, lr, test_d_traid):
        for epoch in tqdm(range(epochs), desc="Epochs:"):
            collector = []
            # 1. 从服务端获得参数
            s_params = self.server.params if epoch != 0 else self._model.state_dict()
            # 参数解密 pending...
            for client_id, client in tqdm(self.clients, desc="Client training"):
                # 2. 用户本地训练产生新参数
                u_params = client.fit(
                    s_params, self.loss_fn, self.optimizer, lr)
                collector.append(u_params)
            # 3. 服务端根据参数更新模型
            self.server.upgrade(collector)

            if (epoch+1) % 1 == 0:
                s_params = self.server.params
                for client_id, client in tqdm(self.clients, desc="Client uploading features"):
                    self.clients.clients_feature_map[client_id] = client.upload_feature(s_params)
                y_list,y_pred_list = self.predict(test_d_traid)

                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                print(f"mae:{mae_},mse:{mse_},rmse:{rmse_}")


    def predict(self, d_traid, similarity_th=0.6,w=0.6):
        s_params = self.server.params
        self._model.load_state_dict(s_params)
        y_pred_list = []
        y_list = []
        traid, p_traid = split_d_traid(d_traid)
        similarity_matrix = self.clients.get_similarity_matrix()

        def upcc():

            total_similarity = 0
            up = 0
            for idx,val in enumerate(similarity_matrix[uid]):
                if uid == idx or val < similarity_th:
                    continue
                up += val * (self.clients.query_rate(idx) - self.clients.query_mean(idx))
                total_similarity += val
            if total_similarity != 0:
                return up / total_similarity
            else:
                return 0

        with torch.no_grad():
            # for batch_id, batch in tqdm(enumerate(test_loader)):
            for traid_row, p_traid_row in tqdm(zip(traid, p_traid),desc="Predict"):

                uid, iid, rate = int(traid_row[0]), int(traid_row[1]), float(traid_row[2])
                user = torch.tensor(np.array(p_traid_row[0])[np.newaxis, :])
                item = torch.tensor(np.array(p_traid_row[1])[np.newaxis, :])
                y_pred = self._model(user, item).item()

                sim = upcc()
                sim_pred = w * y_pred + (1-w) * sim
                # print(f"y_pred:{y_pred}, using sim {sim_pred}, y_real:{rate}")

                y_pred_list.append(sim_pred)
                y_list.append(rate)

import copy
from collections import OrderedDict, defaultdict
from threading import stack_size

import numpy as np
import torch
from data import ToTorchDataset
from models.base import ClientBase
from torch._C import ScriptFunction
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.model_util import (nonzero_mean, split_d_traid, traid_to_matrix,
                              use_optimizer)


class Client(ClientBase):
    """客户端实体
    """
    def __init__(self, traid, uid, model, device, batch_size=-1) -> None:
        """客户端调用fit进行训练

        Args:
            traid: 三元组
            batch_size : local 训练的bs, 默认10, -1表示
        """

        super().__init__(device, model)
        self.traid = traid
        self.uid = uid
        self.loss_list = []
        self.n_item = len(traid)
        self.batch_size = batch_size if batch_size != -1 else self.n_item
        self.data_loader = DataLoader(ToTorchDataset(self.traid),
                                      batch_size=self.batch_size)

    def fit(self, params, loss_fn, optimizer: str, lr, epochs=2):
        return super().fit(params, loss_fn, optimizer, lr, epochs=epochs)

    def __repr__(self) -> str:
        return f"Client(uid={self.uid})"


class Clients(object):
    """多client 的虚拟管理节点
    """
    def __init__(self, traid, model, device) -> None:
        super().__init__()
        self.traid = traid
        self.model = model
        self.device = device
        self.clients_map = {}  # 存储每个client的数据集
        self.client_nums_map = {}

        self._get_clients()

    def sample_clients(self, fraction):
        """Select some fraction of all clients."""
        num_clients = len(self.clients_map)
        num_sampled_clients = max(int(fraction * num_clients), 1)
        sampled_client_indices = sorted(
            np.random.choice(a=[k for k, v in self.clients_map.items()],
                             size=num_sampled_clients,
                             replace=False).tolist())
        return sampled_client_indices

    def _get_clients(self):
        r = defaultdict(list)
        for traid_row in self.traid:
            uid, iid, rate = int(traid_row[0]), int(traid_row[1]), float(
                traid_row[2])
            r[uid].append(traid_row)
        for uid, rows in tqdm(r.items(), desc="Building clients..."):
            self.clients_map[uid] = Client(rows, uid,
                                           copy.deepcopy(self.model),
                                           self.device)
            self.client_nums_map[uid] = len(rows)
        print(f"Clients Nums:{len(self.clients_map)}")
        print(f"Nums for client:", self.client_nums_map)

    def __len__(self):
        return len(self.clients_map)

    def __iter__(self):
        for item in self.clients_map.items():
            yield item

    def __getitem__(self, uid):
        return self.clients_map[uid]

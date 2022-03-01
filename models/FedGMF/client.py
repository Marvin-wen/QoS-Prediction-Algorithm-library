import copy
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from data import ToTorchDataset
from models.base import ClientBase, ClientsBase
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.model_util import (nonzero_user_mean, split_d_traid, traid_to_matrix,
                              use_optimizer)


class Client(ClientBase):
    """客户端实体
    """
    def __init__(self, traid, uid, model, device, batch_size=32) -> None:
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

    def fit(self, params, loss_fn, optimizer: str, lr, epochs=5):
        return super().fit(params, loss_fn, optimizer, lr, epochs=epochs)

    def __repr__(self) -> str:
        return f"Client(uid={self.uid})"


class Clients(ClientsBase):
    """多client 的虚拟管理节点
    """
    def __init__(self, traid, model, device) -> None:
        self.client_nums_map = {}
        super().__init__(traid, model, device)

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

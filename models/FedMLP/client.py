from collections import OrderedDict, defaultdict

import torch
from tqdm import tqdm
import numpy as np
from data import ToTorchDataset
from torch.utils.data import DataLoader
from utils.model_util import split_d_traid, nonzero_mean, traid_to_matrix, use_optimizer
import copy

class Client(object):
    """客户端实体
    """

    def __init__(self, traid, uid, model,device) -> None:
        super().__init__()
        self.traid = traid
        self.device = device
        self.uid = uid
        self.model = model
        self.loss_list = []
        self.n_item = len(traid)
        self.train_loader = DataLoader(
            ToTorchDataset(self.traid), batch_size=10)

    def fit(self, params, loss_fn, optimizer, lr, epoch=5):
        total_loss = 0
        self.model.load_state_dict(params)
        self.model.train()
        opt = optimizer(self.model.parameters(), lr)
        for i in range(epoch):
            train_batch_loss = 0
            for batch_id, batch in enumerate(self.train_loader):
                user, item, rating = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_real = rating.reshape(-1, 1)
                opt.zero_grad()
                y_pred = self.model(user, item)
                loss = loss_fn(y_pred, y_real)
                loss.backward()
                opt.step()
                train_batch_loss += loss.item()

            loss_per_epoch = train_batch_loss / len(self.train_loader)
            self.loss_list.append(loss_per_epoch)
            total_loss += loss_per_epoch
            
        return self.model.state_dict(), round(total_loss/epoch,4)


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

    def _get_clients(self):
        r = defaultdict(list)
        for traid_row in self.traid:
            uid, iid, rate = int(traid_row[0]), int(
                traid_row[1]), float(traid_row[2])
            r[uid].append(traid_row)
        for uid, rows in tqdm(r.items(), desc="Building clients..."):
            self.clients_map[uid] = Client(rows, uid, copy.deepcopy(self.model), self.device)
            self.client_nums_map[uid] = len(rows)
        print(f"Clients Nums:{len(self.clients_map)}")
        print(f"Nums for client:",self.client_nums_map)


    def __len__(self):
        return len(self.clients_map)

    def __iter__(self):
        for item in self.clients_map.items():
            yield item

    def __getitem__(self, uid):
        return self.clients_map[uid]

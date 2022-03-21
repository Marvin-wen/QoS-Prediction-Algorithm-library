import copy
from collections import OrderedDict, defaultdict
from functools import partialmethod

import numpy as np
import torch
from data import ToTorchDataset
from models.base import ClientBase
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.model_util import (nonzero_user_mean, split_d_triad,
                              triad_to_matrix, use_optimizer)


class Client(ClientBase):
    """客户端实体
    """
    def __init__(self,
                 triad,
                 uid,
                 device,
                 model,
                 batch_size=-1,
                 local_epochs=5) -> None:
        super().__init__(device, model)
        self.triad = triad
        self.uid = uid
        self.n_item = len(triad)
        self.local_epochs = local_epochs
        self.batch_size = self.n_item if batch_size == -1 else batch_size
        self.data_loader = DataLoader(ToTorchDataset(self.triad),
                                      batch_size=self.batch_size,
                                      drop_last=True)
        self.single_batch = DataLoader(ToTorchDataset(self.triad),
                                       batch_size=1,
                                       drop_last=True)

    def fit(self, params, loss_fn, optimizer: str, lr, epochs=5):
        return super().fit(params, loss_fn, optimizer, lr, epochs=epochs)

    def upload_feature(self, params):
        self.model.load_state_dict(params)
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(self.single_batch):
                user, item, rating = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_pred, u_feature, i_feature = self.model(user, item, True)
                return u_feature[0]


class Clients(object):
    """多client 的虚拟管理节点
    """
    def __init__(self,
                 d_triad,
                 model,
                 device,
                 batch_size=-1,
                 local_epochs=5) -> None:
        super().__init__()
        self.triad, self.p_triad = split_d_triad(d_triad)
        self.model = model
        self.device = device
        self.clients_map = {}  # 存储每个client的数据集
        self.clients_feature_map = OrderedDict()  # 存储每个client的feature
        self.triad2matrix = triad_to_matrix(self.triad, -1)
        self.u_mean = nonzero_user_mean(self.triad2matrix, -1)
        self.batch_size = batch_size
        self.local_epochs = local_epochs

        self._get_clients()

    def _get_clients(self):
        r = defaultdict(list)
        for triad_row, p_triad_row in zip(self.triad, self.p_triad):
            uid, iid, rate = int(triad_row[0]), int(triad_row[1]), float(
                triad_row[2])
            r[uid].append(p_triad_row)
        for uid, rows in tqdm(r.items(), desc="Building clients..."):
            self.clients_map[uid] = Client(rows,
                                           uid,
                                           self.device,
                                           copy.deepcopy(self.model),
                                           batch_size=self.batch_size,
                                           local_epochs=self.local_epochs)
        print(f"Clients Nums:{len(self.clients_map)}")

    def sample_clients(self, fraction):
        """Select some fraction of all clients."""
        num_clients = len(self.clients_map)
        num_sampled_clients = max(int(fraction * num_clients), 1)
        sampled_client_indices = sorted(
            np.random.choice(a=[k for k, v in self.clients_map.items()],
                             size=num_sampled_clients,
                             replace=False).tolist())
        return sampled_client_indices

    def get_similarity_matrix(self):
        l = []
        cnt = 0  # 这个是为了防止user取得不连续
        for uid, val in OrderedDict(
                sorted(self.clients_feature_map.items(),
                       key=lambda x: x[0])).items():
            if cnt != int(uid):
                for i in range(int(uid) - cnt):
                    l.append(np.zeros_like(val.numpy()))
                cnt = uid
            else:
                l.append(val.cpu().numpy())
                cnt += 1
        l = np.array(l)
        l = np.corrcoef(l)
        return l

    def _query(self, uid, iid, type_="rate"):
        try:
            if type_ == "rate":
                return self.triad2matrix[uid][iid]
            elif type_ == "mean":
                return self.u_mean[uid]
        except Exception:
            return None

    # https://stackoverflow.com/questions/16626789/functools-partial-on-class-method
    # The code below is wrong!
    # def query_rate(self):
    #     return partial(self._query, self=self, type_="rate")

    def query_rate(self, uid, iid):
        return self._query(uid, iid, "rate")

    def query_mean(self, uid):
        return self._query(uid, None, "mean")

    def __len__(self):
        return len(self.clients_map)

    def __iter__(self):
        for item in self.clients_map.items():
            yield item

    def __getitem__(self, uid):
        return self.clients_map[uid]

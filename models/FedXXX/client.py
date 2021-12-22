from collections import OrderedDict, defaultdict
from functools import partialmethod

import torch
from tqdm import tqdm
import numpy as np
from data import ToTorchDataset
from torch.utils.data import DataLoader
from utils.model_util import split_d_traid, nonzero_mean, traid_to_matrix


class Client(object):
    """客户端实体
    """

    def __init__(self, traid, uid, device, model) -> None:
        super().__init__()
        self.traid = traid
        self.uid = uid
        self.device = device
        self.model = model
        self.n_item = len(traid)
        self.train_loader = DataLoader(
            ToTorchDataset(self.traid), batch_size=128, drop_last=True)
        self.single_batch = DataLoader(
            ToTorchDataset(self.traid), batch_size=1, drop_last=True)

    def fit(self, params, loss_fn, optimizer, lr):
        self.model.load_state_dict(params)
        opt = optimizer(self.model.parameters(), lr)
        for batch_id, batch in enumerate(self.train_loader):
            user, item, rating = batch[0].to(self.device), batch[1].to(
                self.device), batch[2].to(self.device)
            # print(user, item, rating)
            y_real = rating.reshape(-1, 1)
            opt.zero_grad()
            y_pred = self.model(user, item)
            loss = loss_fn(y_pred, y_real)
            loss.backward()
            opt.step()
        return self.model.state_dict(),loss

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

    def __init__(self, d_traid, model, use_gpu=True) -> None:
        super().__init__()
        self.traid, self.p_traid = split_d_traid(d_traid)
        self.model = model
        self.device = ("cuda" if (
            use_gpu and torch.cuda.is_available()) else "cpu")
        if use_gpu:
            self.model.to(self.device)
        self.clients_map = {}  # 存储每个client的数据集
        self.clients_feature_map = OrderedDict()  # 存储每个client的feature
        self.traid2matrix = traid_to_matrix(self.traid, -1)
        self.u_mean = nonzero_mean(self.traid2matrix, -1)

        self._get_clients()

    def _get_clients(self):
        r = defaultdict(list)
        for traid_row, p_traid_row in zip(self.traid, self.p_traid):
            uid, iid, rate = int(traid_row[0]), int(
                traid_row[1]), float(traid_row[2])
            r[uid].append(p_traid_row)
        for uid, rows in tqdm(r.items(), desc="Building clients..."):
            self.clients_map[uid] = Client(rows, uid, self.device, self.model)
        print(f"Clients Nums:{len(self.clients_map)}")

    def get_similarity_matrix(self):
        l = []
        cnt = 0  # 这个是为了防止user取得不连续
        for uid, val in OrderedDict(sorted(self.clients_feature_map.items(), key=lambda x: x[0])).items():
            if cnt != int(uid):
                for i in range(int(uid)-cnt):
                    l.append(np.zeros_like(val.numpy()))
                cnt = uid
            else:
                l.append(val.cpu().numpy())
                cnt += 1
        l = np.array(l)
        return l

    def _query(self, uid, iid, type_="rate"):
        try:
            if type_ == "rate":
                return self.traid2matrix[uid][iid]
            elif type_ == "mean":
                return self.u_mean[uid]
        except Exception:
            return None
    # https://stackoverflow.com/questions/16626789/functools-partial-on-class-method 
    # The code below is wrong!
    # def query_rate(self):
    #     return partial(self._query, self=self, type_="rate")



    def query_rate(self,uid,iid):
        return self._query(uid,iid,"rate")

    def query_mean(self,uid):
        return self._query(uid,None,"mean")

    def __len__(self):
        return len(self.clients_map)

    def __iter__(self):
        for item in self.clients_map.items():
            yield item

    def __getitem__(self, uid):
        return self.clients_map[uid]

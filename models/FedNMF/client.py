from collections import defaultdict

import numpy as np
from tqdm import tqdm


class Client(object):
    def __init__(self, triad, uid, user_vec) -> None:
        super().__init__()
        self.triad = triad
        self.uid = uid
        self.n_item = len(triad)
        self.user_vec = user_vec

    def fit(self, items_vec, lr):
        l = []
        # 1. 获取服务端传过来的物品特征矩阵
        for row in self.triad:
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            # 获得预测值
            y_pred = self.user_vec @ items_vec[iid].T
            e_ui = rate - y_pred
            # 2. 根据物品特征矩阵计算用户和物品梯度
            user_grad = -1 * e_ui * items_vec[iid]
            item_grad = -1 * e_ui * user_grad
            # 3. 用户梯度更新用户特征矩阵
            self.user_vec -= lr * user_grad
            self.user_vec[self.user_vec < 0] = 0
            # 4. 物品梯度返回
            l.append([iid, item_grad])
        return l


class Clients(object):
    def __init__(self, triad, n_user, latent_dim) -> None:
        super().__init__()
        self.triad = triad
        self.clients_map = {}
        # self.users_vec = 2 * np.random.random((n_user, latent_dim)) - 1
        self.users_vec = np.random.random((n_user, latent_dim))
        self._get_clients()

    def _get_clients(self):
        r = defaultdict(list)
        for row in self.triad:
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            r[uid].append(row)
        for uid, rows in r.items():
            self.clients_map[uid] = Client(np.array(rows), uid,
                                           self.users_vec[uid])
        print(f"Clients Nums:{len(self.clients_map)}")

    def __len__(self):
        return len(self.clients_map)

    def __iter__(self):
        for item in self.clients_map.items():
            yield item

    def __getitem__(self, uid):
        return self.clients_map[uid]

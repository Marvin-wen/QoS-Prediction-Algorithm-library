import copy

import numpy as np
# Non-negative Matrix Factorization
from sklearn.decomposition import NMF
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse
from utils.model_util import triad_to_matrix


class NMFModel(object):
    """Non-negative Matrix Factorization Model
    """
    def __init__(self, n_user, n_item, latent_dim) -> None:
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.latent_dim = latent_dim  # 隐特征维度
        self.user_matrix = None  # 用户特征矩阵
        self.item_matrix = None  # 项目特征矩阵
        self.matrix = None  # 真实矩阵
        self._nan_symbol = -1

    def _init_matrix(self, traid):
        """初始化用户和项目的特征矩阵
        """
        # self.user_matrix = 2 * np.random.random((self.n_user, self.latent_dim)) - 1
        # self.item_matrix = 2 * np.random.random((self.latent_dim, self.n_item)) - 1

        self.user_matrix = np.random.random((self.n_user, self.latent_dim))
        self.item_matrix = np.random.random((self.latent_dim, self.n_item))
        self.matrix = triad_to_matrix(traid, self._nan_symbol)
        self.matrix[self.matrix == self._nan_symbol] = 0

    def _normalize(self):
        m, n = self.matrix.shape
        _matrix = copy.deepcopy(self.matrix)
        for i in range(n):
            self.matrix[:, i] = _matrix[:, i] / np.sum(_matrix[:, i])

    def fit(self,
            traid,
            test,
            epochs=100,
            verbose=False,
            early_stop=True,
            normalize=False):
        if (not self.user_matrix and not self.item_matrix) or self.matrix:
            self._init_matrix(traid)

        # 测试了貌似没效果
        if normalize:
            self._normalize()
        """
        # FIXME 手动实现，有时会不收敛，没找到问题
        eps = 0.00001
        for epoch in tqdm(range(epochs), desc="NMF Training Epoch"):
            W = copy.deepcopy(self.user_matrix)
            H = copy.deepcopy(self.item_matrix)
            self.user_matrix = W * (self.matrix @ H.T) / (W @ H @ H.T + eps)
            W = copy.deepcopy(self.user_matrix)
            H = copy.deepcopy(self.item_matrix)
            self.item_matrix = H * (W.T @ self.matrix) / (W.T @ W @ H + eps)

        
        # sklearn实现
        # nmf_model = NMF(n_components=self.latent_dim, init='nndsvda', solver='mu')
        # self.user_matrix = nmf_model.fit_transform(self.matrix)
        # self.item_matrix = nmf_model.components_

        """

        for epoch in tqdm(range(epochs), desc="NMF Training Epoch"):

            tmp_user_matrix = copy.deepcopy(self.user_matrix)
            tmp_item_matrix = copy.deepcopy(self.item_matrix)

            up_W = self.matrix @ self.item_matrix.T
            down_W = self.user_matrix @ self.item_matrix @ self.item_matrix.T

            self.user_matrix = self.user_matrix * up_W / down_W

            up_H = self.user_matrix.T @ self.matrix
            down_H = self.user_matrix.T @ self.user_matrix @ self.item_matrix + eps

            self.item_matrix = self.item_matrix * up_H / down_H


            if early_stop and np.mean(np.abs(self.user_matrix - tmp_user_matrix)) < 1e-4 and \
                np.mean(np.abs(self.item_matrix - tmp_item_matrix)) < 1e-4:
                print('Converged')
                break

            if verbose and (epoch + 1) % 200 == 0:
                y_list, y_pred_list = self.predict(test)
                print(f"[{epoch}/{epochs}] MAE:{mae(y_list,y_pred_list):.5f}")

    def predict(self, traid):
        assert self.user_matrix is not None, "Please fit first e.g. model.fit()"
        y_pred_list = []
        y_list = []
        for row in tqdm(traid, desc='NMF Perdicting'):
            uid, iid, y = int(row[0]), int(row[1]), float(row[2])
            y_pred = self.user_matrix[uid] @ self.item_matrix[:, iid]
            y_pred_list.append(y_pred)
            y_list.append(y)
        return y_list, y_pred_list

import copy

import numpy as np
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse


class MFModel(object):
    def __init__(self, n_user, n_item, latent_dim, lr, lambda_) -> None:
        super().__init__()
        self.lr = lr
        self.lambda_ = lambda_
        self.latent_dim = latent_dim
        self.n_user = n_user
        self.n_item = n_item
        self.user_vec = None
        self.item_vec = None

    def _init_vec(self):
        """初始化用户和物品的特征矩阵
        """
        self.user_vec = 2 * np.random.random(
            (self.n_user, self.latent_dim)) - 1
        self.item_vec = 2 * np.random.random(
            (self.n_item, self.latent_dim)) - 1

    def fit(self, traid, test, epochs=100, verbose=True, early_stop=True):
        if not self.user_vec and not self.item_vec:
            self._init_vec()

        for epoch in tqdm(range(epochs), desc="MF Training Epoch"):

            tmp_user_vec = copy.deepcopy(self.user_vec)
            tmp_item_vec = copy.deepcopy(self.item_vec)

            for row in traid:
                user_idx, item_idx, y = int(row[0]), int(row[1]), float(row[2])
                y_pred = self.user_vec[user_idx] @ self.item_vec[item_idx].T
                e_ui = y - y_pred
                user_grad = -2 * e_ui * self.item_vec[
                    item_idx] + 2 * self.lambda_ * self.user_vec[user_idx]
                item_grad = -2 * e_ui * self.user_vec[
                    user_idx] + 2 * self.lambda_ * self.item_vec[item_idx]
                self.user_vec[user_idx] -= self.lr * user_grad
                self.item_vec[item_idx] -= self.lr * item_grad

            if early_stop and np.mean(np.abs(self.user_vec - tmp_user_vec)) < 1e-4 and \
                np.mean(np.abs(self.item_vec - tmp_item_vec)) < 1e-4:
                print('Converged')
                break

            if verbose and (epoch + 1) % 10 == 0:
                y_list, y_pred_list = self.predict(test)
                print(f"[{epoch}/{epochs}] MAE:{mae(y_list,y_pred_list):.5f}")

    def predict(self, traid):
        assert isinstance(self.user_vec,
                          np.ndarray), "please fit first e.g. model.fit()"
        y_pred_list = []
        y_list = []
        for row in tqdm(traid):
            uid, iid, y = int(row[0]), int(row[1]), float(row[2])
            y_pred = self.user_vec[uid] @ self.item_vec[iid].T
            y_pred_list.append(y_pred)
            y_list.append(y)
        return y_list, y_pred_list

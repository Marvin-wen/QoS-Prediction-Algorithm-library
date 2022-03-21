import os

import numpy as np
from root import absolute
from tqdm import tqdm
from utils import TNLog
from utils.evaluation import mae, mse, rmse

from .client import Clients
from .server import Server


class FedMF(object):
    def __init__(self, server: Server, clients: Clients) -> None:
        super().__init__()
        self.server = server
        self.clients = clients
        self.name = __class__.__name__
        self.logger = TNLog(self.name)
        self.logger.initial_logger()

    def fit(self, epochs, lambda_, lr, test_triad, scaler=None):
        best_mae = None
        is_better = True
        for epoch in tqdm(range(epochs), desc="Epochs"):
            gradient_from_user = []
            # 遍历每一个用户
            for client_id, client in self.clients:
                # client upgrade
                gradient_from_user.extend(
                    client.fit(self.server.items_vec, lambda_, lr))

            # server upgrade
            self.server.upgrade(lr, gradient_from_user)

            if (epoch + 1) % 200 == 0:
                y_list, y_pred_list = self.predict(test_triad, scaler=scaler)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)
                if best_mae is None or mae_ < best_mae:
                    best_mae = mae_
                    is_better = True
                else:
                    is_better = False
                self.save_checkpoint(is_better,
                                     f"epoch_{epoch+1}_mae_{mae_:.4f}")
                self.logger.info(
                    f"Epoch:{epoch+1} mae:{mae_},mse:{mse_},rmse:{rmse_}")

    def save_checkpoint(self, is_better, prefix=""):
        if is_better == True:
            file_path = absolute(f"output/{self.name}")
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            user_vec_filepath = file_path + f"/{prefix}_users_vec.npy"
            item_vec_filepath = file_path + f"/{prefix}_items_vec.npy"
            np.save(user_vec_filepath, self.clients.users_vec)
            print(f"Save user vector=>{user_vec_filepath}")
            np.save(item_vec_filepath, self.server.items_vec)
            print(f"Save item vector=>{item_vec_filepath}")
        else:
            print("Not improved")

    def load_checkpoint(self, user_vec_path, item_vec_path):
        users_vec = np.load(user_vec_path)
        items_vec = np.load(item_vec_path)
        return users_vec, items_vec

    def predict(self,
                triad,
                resume=False,
                user_vec_path=None,
                item_vec_path=None,
                scaler=None):

        y_list = []
        y_pred_list = []

        if resume:
            users_vec, items_vec = self.load_checkpoint(
                user_vec_path, item_vec_path)
            for row in tqdm(triad, desc="Test"):
                uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
                y_pred = users_vec[uid] @ items_vec[iid].T
                y_list.append(rate)
                y_pred_list.append(y_pred)
        else:
            for row in tqdm(triad, desc="Test"):
                uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
                y_pred = self.clients[uid].user_vec @ self.server.items_vec[
                    iid].T
                y_list.append(rate)
                y_pred_list.append(y_pred)

        if scaler is not None:
            y_list = np.array(y_list, dtype=np.float)
            y_pred_list = np.array(y_pred_list, dtype=np.float)

            y_list = scaler.inverse_transform(y_list)
            y_pred_list = scaler.inverse_transform(y_pred_list)

        return y_list, y_pred_list

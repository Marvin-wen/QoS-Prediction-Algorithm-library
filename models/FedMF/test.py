from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from . import Clients, Server
from .model import FedMF

"""
RESULT FedMF: 
100epoch
Density:0.05,type:rt,mae:0.6702588330384307,mse:2.442865720480839,rmse:1.5629669607771108
Density:0.1,type:rt,mae:0.5168279573942084,mse:1.722897863020363,rmse:1.3125920398282032
Density:0.15,type:rt,mae:0.47297485096486525,mse:1.5074749994329864,rmse:1.227792734720721
"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "rt"
    latent_dim = 8
    lr = 0.001
    lambda_ = 0.1
    epochs = 10
    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)

    clients = Clients(train_data, md_data.row_n, latent_dim)
    server = Server(md_data.col_n, latent_dim)

    mf = FedMF(server, clients)
    mf.fit(epochs, lambda_, lr)
    y, y_pred = mf.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

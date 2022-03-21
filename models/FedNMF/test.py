from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from . import Clients, Server
from .model import FedNMF
"""
RESULT FedNMF: 
1000epoch

"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    # 1
    type_ = "rt"
    latent_dim = 8
    lr = 0.01
    epochs = 1000

    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)
    clients = Clients(train_data, md_data.row_n, latent_dim)

    server = Server(md_data.col_n, latent_dim)

    nmf = FedNMF(server, clients)
    nmf.fit(epochs, lr, test_data)
    y, y_pred = nmf.predict(test_data, True, None)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

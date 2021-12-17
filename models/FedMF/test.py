from .import Clients
from .import Server
from data import MatrixDataset
from .model import FedMF
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

"""
RESULT FedMF:

"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "rt"
    latent_dim = 8
    lr = 0.001
    lambda_ = 0.1
    epochs = 100
    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)

    clients = Clients(train_data, md_data.row_n, latent_dim)
    server = Server(md_data.col_n, latent_dim)

    mf = FedMF(server, clients.clients_map)
    mf.fit(epochs, lambda_, lr)
    y, y_pred = mf.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

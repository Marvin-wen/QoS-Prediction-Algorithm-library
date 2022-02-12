from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from . import Clients, Server
from .model import FedMF
"""
RESULT FedMF: 
1000epoch
Density:0.05,type:rt,mae:0.6090604947629363,mse:2.3006280931616536,rmse:1.5167821508580768
Density:0.1,type:rt,mae:0.5071641017174705,mse:1.7203263210368958,rmse:1.3116121076891962
Density:0.15,type:rt,mae:0.46475316376452325,mse:1.4854062714808631,rmse:1.2187724445034287
Density:0.2,type:rt,mae:0.43765304163567553,mse:1.3690546770840173,rmse:1.1700660994508034

"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    # 1
    type_ = "tp"
    latent_dim = 8
    lr = 0.001
    lambda_ = 0.1
    epochs = 1000

    # 2
    # type_ = "rt"
    # latent_dim = 12
    # lr = 0.0005
    # lambda_ = 0.1
    # epochs = 2000

    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density,
                                                     normalize_type="z_score")
    clients = Clients(train_data, md_data.row_n, latent_dim)

    server = Server(md_data.col_n, latent_dim)

    mf = FedMF(server, clients)
    mf.fit(epochs, lambda_, lr, test_data, scaler=md_data.scaler)
    y, y_pred = mf.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

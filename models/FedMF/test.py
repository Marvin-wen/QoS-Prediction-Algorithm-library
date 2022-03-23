from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from . import Clients, Server
from .model import FedMF

from utils.mylogger import TNLog

"""
RESULT FedMF: 
1000epoch
Density:0.05,type:rt,mae:0.5949614110683237,mse:2.1892629584652785,rmse:1.4796158144820157
Density:0.1,type:rt,mae:0.5065232463307372,mse:1.6816029233716594,rmse:1.2967663333737731
Density:0.15,type:rt,mae:0.466420129810779,mse:1.4792783323010215,rmse:1.216255866296653
Density:0.2,type:rt,mae:0.43765304163567553,mse:1.3690546770840173,rmse:1.1700660994508034

# 1 FedMF 优化后的结果


# 2 FedMF TP结果

"""

logger = TNLog('FedMF')
logger.initial_logger()

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    # 1
    type_ = "tp"
    latent_dim = 8
    lr = 0.0001
    lambda_ = 0.1
    epochs = 3000

    # 2
    # type_ = "rt"
    # latent_dim = 12
    # lr = 0.0005
    # lambda_ = 0.1
    # epochs = 2000

    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)
    clients = Clients(train_data, md_data.row_n, latent_dim)

    server = Server(md_data.col_n, latent_dim)

    mf = FedMF(server, clients)
    mf.fit(epochs, lambda_, lr, test_data, scaler=md_data.scaler)
    y, y_pred = mf.predict(test_data, False)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    logger.info(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

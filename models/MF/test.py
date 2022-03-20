from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from model import MFModel
# 日志
from utils.mylogger import TNLog

"""
RESULT MF:
Density:0.05, type:rt, mae:0.6717, mse:2.4526, rmse:1.5661
Density:0.10, type:rt, mae:0.5169, mse:1.7230, rmse:1.3126
Density:0.15, type:rt, mae:0.4725, mse:1.5054, rmse:1.2269
Density:0.20, type:rt, mae:0.4396, mse:1.3708, rmse:1.1708
"""

freeze_random()  # 冻结随机数 保证结果一致

logger = TNLog('MF')
logger.initial_logger()

for density in [0.05, 0.1, 0.15, 0.2]:
    type_ = "tp"
    latent_dim = 8
    lr = 0.0001
    lambda_ = 0.05
    epochs = 200
    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)

    mf = MFModel(md_data.row_n, md_data.col_n, latent_dim, lr, lambda_)
    mf.fit(train_data, test_data, epochs, verbose=False)
    y, y_pred = mf.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    logger.info(f"Density:{density:.02f}, type:{type_}, mae:{mae_:.04f}, mse:{mse_:.04f}, rmse:{rmse_:.04f}")

from data import MatrixDataset
# Non-negative Matrix Factorization
from sklearn.decomposition import NMF
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random
# 日志
from utils.mylogger import TNLog

from .model import NMFModel

"""
RESULT NMF:
Density:0.05, type:rt, latent_dim:  8, epochs: 200, mae:0.6715, mse:2.3325, rmse:1.5272
Density:0.10, type:rt, latent_dim:  8, epochs: 200, mae:0.5599, mse:1.9375, rmse:1.3919
Density:0.15, type:rt, latent_dim:  8, epochs: 200, mae:0.5094, mse:1.7087, rmse:1.3072
"""

freeze_random()  # 冻结随机数 保证结果一致

logger = TNLog('NMF')
logger.initial_logger()

for density in [0.05, 0.1, 0.15, 0.2]:
    type_ = "rt"
    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)

    # latent_dim = 50
    # epochs = 1000
    # for latent_dim in range(2, 31, 4):
    #     for epochs in range(10, 100, 20):
    #         nmf = NMFModel(md_data.row_n, md_data.col_n, latent_dim)
    #         nmf.fit(train_data, test_data, epochs, verbose=False)
    #         y, y_pred = nmf.predict(test_data)

    #         mae_ = mae(y, y_pred)
    #         mse_ = mse(y, y_pred)
    #         rmse_ = rmse(y, y_pred)

    #         logger.info(
    #             f"Density:{density:.02f}, type:{type_}, latent_dim:{latent_dim:{3}}, epochs:{epochs:{4}}, mae:{mae_:.04f}, mse:{mse_:.04f}, rmse:{rmse_:.04f}")

    latent_dim = 8
    epochs = 200
    nmf = NMFModel(md_data.row_n, md_data.col_n, latent_dim)
    nmf.fit(train_data, test_data, epochs, verbose=True, normalize=False,early_stop=True)
    y, y_pred = nmf.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    logger.info(
        f"Density:{density:.02f}, type:{type_}, latent_dim:{latent_dim:{3}}, epochs:{epochs:{4}}, mae:{mae_:.04f}, mse:{mse_:.04f}, rmse:{rmse_:.04f}"
    )

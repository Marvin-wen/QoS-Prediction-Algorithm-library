from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from models.UIPCC import UIPCCModel

# 冻结随机数
from utils.model_util import freeze_random
# 日志
from utils.mylogger import TNLog

"""
RESULT UIPCC:
Density:0.05, type:rt, lambda:0.8, mae:0.6398, mse:2.1733, rmse:1.4742
Density:0.10, type:rt, lambda:0.8, mae:0.5360, mse:1.8121, rmse:1.3461
Density:0.15, type:rt, lambda:0.8, mae:0.4876, mse:1.6138, rmse:1.2704
Density:0.20, type:rt, lambda:0.8, mae:0.4608, mse:1.4924, rmse:1.2216

Density:0.05, type:tp, lambda:0.8, mae:29.7889, mse:5134.6681, rmse:71.6566
Density:0.10, type:tp, lambda:0.8, mae:22.8508, mse:3772.6413, rmse:61.4218
Density:0.15, type:tp, lambda:0.8, mae:19.5598, mse:2985.7650, rmse:54.6422
Density:0.20, type:tp, lambda:0.8, mae:17.8447, mse:2536.0946, rmse:50.3597
"""

freeze_random()  # 冻结随机数 保证结果一致

logger = TNLog('UIPCCModel')
logger.initial_logger()

for density in [0.05, 0.1, 0.15, 0.2]:
    type_ = "rt"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    uipcc = UIPCCModel()
    uipcc.fit(train_data)

    lamb = 0.8
    y, y_pred = uipcc.predict(test_data, 30, 500, lamb)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    logger.info(
        f"Density:{density:.02f}, type:{type_}, lambda:{lamb}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}")

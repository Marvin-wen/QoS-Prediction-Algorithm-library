from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from model import UIPCCModel

# 冻结随机数
from utils.model_util import freeze_random
# 日志
from utils.mylogger import TNLog

"""
RESULT UIPCC:
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

    y, y_pred = uipcc.predict(test_data, 20, 500, 0.2)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    logger.info(f"Density:{density:.02f}, type:{type_}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}")

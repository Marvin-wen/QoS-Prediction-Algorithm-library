from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from model import IPCCModel
# 冻结随机数
from utils.model_util import freeze_random
# 日志
from utils.mylogger import TNLog
# 多进程
import multiprocessing as mp

"""
RESULT IPCC:
Density:0.05, topk:200, type:rt, mae:0.8413, mse:3.6662, rmse:1.9147
"""

freeze_random()  # 冻结随机数 保证结果一致

logger = TNLog('IPCCModel')
logger.initial_logger()

for density in [0.05, 0.1, 0.15, 0.2]:
    type_ = "rt"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    imean = IPCCModel()
    imean.fit(train_data, metric='PCC')

    for topk in [50, 100, 200, 500]:
        y, y_pred = imean.predict(test_data, topk)

        mae_ = mae(y, y_pred)
        mse_ = mse(y, y_pred)
        rmse_ = rmse(y, y_pred)

        logger.info(f"Density:{density}, topk:{topk}, type:{type_}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}")

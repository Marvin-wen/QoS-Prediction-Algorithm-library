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
Density:0.05, topk:1000, type:rt, mae:0.7027, mse:2.0583, rmse:1.4347
Density:0.1,  topk:1000, type:rt, mae:0.6291, mse:1.8175, rmse:1.3481
Density:0.15, topk:1000, type:rt, mae:0.5524, mse:1.6258, rmse:1.2751
Density:0.2,  topk:1000, type:rt, mae:0.5150, mse:1.5396, rmse:1.2408

Density:0.05, type:tp, topK:500, mae:29.4089, mse:4340.1094, rmse:65.8795
Density:0.1,  type:tp, topK:500, mae:28.9753, mse:3893.7472, rmse:62.3999
Density:0.15, type:tp, topK:500, mae:27.2390, mse:3382.6660, rmse:58.1607
Density:0.2,  type:tp, topK:500, mae:26.8617, mse:3204.2192, rmse:56.6058
"""

freeze_random()  # 冻结随机数 保证结果一致

logger = TNLog('IPCCModel')
logger.initial_logger()

for density in [0.05, 0.1, 0.15, 0.2]:
    type_ = "tp"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    imean = IPCCModel()
    imean.fit(train_data, metric='PCC')

    topk = 500
    y, y_pred = imean.predict(test_data, topk)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    logger.info(f"Density:{density}\ttype:{type_}\ttopk:{topk}\tmae:{mae_:.4f}\tmse:{mse_:.4f}\trmse:{rmse_:.4f}")

from cmath import log
from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import UPCCModel
from utils import logger
"""
RESULT UPCC:

Density:0.05,type:rt,mae:0.6989300707521432,mse:2.7737428845756824,rmse:1.6654557588166916
Density:0.1,type:rt,mae:0.5598082846452833,mse:2.1501265216568024,rmse:1.4663309727536966
Density:0.15,type:rt,mae:0.49665146697083623,mse:1.822202449824236,rmse:1.349889791732731
Density:0.2,type:rt,mae:0.4640162396499109,mse:1.6248480178887317,rmse:1.2746952647157406

Density:0.05,type:tp,mae:36.44804703916447,mse:7770.4758688545135,rmse:88.15030271561473
Density:0.1,type:tp,mae:24.80577811244719,mse:4828.352330817265,rmse:69.4863463625572
Density:0.15,type:tp,mae:20.103870591282085,mse:3577.5086548164077,rmse:59.812278462004834
Density:0.2,type:tp,mae:17.75683423278253,mse:2891.0985679440246,rmse:53.76893683107399



"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "tp"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    umean = UPCCModel()
    umean.fit(train_data)
    y, y_pred = umean.predict(test_data, 20)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
    logger.info(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

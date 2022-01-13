from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import IMEANModel
"""
RESULT IMEAN:
Density:0.05,type:rt,mae:0.7036465465739269,mse:2.4719828954178786,rmse:1.5722540810625611
Density:0.1,type:rt,mae:0.688810904993331,mse:2.3660737203153084,rmse:1.5382047068954472
Density:0.15,type:rt,mae:0.6848530991651357,mse:2.3445951209481284,rmse:1.5312070797080741
Density:0.2,type:rt,mae:0.6799561122796471,mse:2.3402777616398236,rmse:1.5297966406159427

Density:0.05,type:tp,mae:27.27003051028569,mse:4370.019426321108,rmse:66.10612245716057
Density:0.1,type:tp,mae:26.89339188891224,mse:4194.168463888392,rmse:64.76240007819655
Density:0.15,type:tp,mae:26.6891851209872,mse:4156.781147853273,rmse:64.47310406559679
Density:0.2,type:tp,mae:26.626194861141634,mse:4106.05625434028,rmse:64.0785163244303
"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:
    type_ = "tp"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    umean = IMEANModel()
    umean.fit(train_data)
    y, y_pred = umean.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

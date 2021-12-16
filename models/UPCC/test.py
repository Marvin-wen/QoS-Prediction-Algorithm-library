from data import MatrixDataset
from .model import UPCCModel
from utils.evaluation import mae,mse,rmse
from utils.model_util import freeze_random

"""
RESULT UPCC:
Density:0.05,type:rt,mae:0.7334384829325753,mse:2.6762234507993745,rmse:1.635916700446381
Density:0.1,type:rt,mae:0.628473598766662,mse:2.1314974642028677,rmse:1.4599648845786901
Density:0.15,type:rt,mae:0.5801255883194202,mse:1.8990928388487136,rmse:1.3780757739865808
Density:0.2,type:rt,mae:0.5585672563669457,mse:1.7809544697289579,rmse:1.3345240611277707

Density:0.05,type:tp,mae:31.433279057978826,mse:5942.681984715335,rmse:77.08879286067031
Density:0.1,type:tp,mae:24.705758868958824,mse:4119.820902493704,rmse:64.18583101038503
Density:0.15,type:tp,mae:22.348603068910084,mse:3475.7766495526553,rmse:58.95571770025919
Density:0.2,type:tp,mae:21.212877869416783,mse:3153.9556887043786,rmse:56.16008982101416
"""

freeze_random() # 冻结随机数 保证结果一致

for density in [0.05,0.1,0.15,0.2]:

    type_ = "rt"
    rt_data = MatrixDataset(type_)
    train_data,test_data = rt_data.split_train_test(density)


    umean = UPCCModel()
    umean.fit(train_data)
    y,y_pred = umean.predict(test_data,20)


    mae_ = mae(y,y_pred)
    mse_ = mse(y,y_pred)
    rmse_ = rmse(y,y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")



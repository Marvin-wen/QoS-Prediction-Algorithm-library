from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import UMEANModel
"""
RESULT UMEAN:
Density:0.05,type:rt,mae:0.8816043906403586,mse:3.4499158310187976,rmse:1.8573949044343794
Density:0.1,type:rt,mae:0.8754038821670327,mse:3.4431827593576285,rmse:1.8555815151476447
Density:0.15,type:rt,mae:0.8769691658302028,mse:3.435824345552405,rmse:1.8535976762912725
Density:0.2,type:rt,mae:0.8770991226703511,mse:3.434068984920858,rmse:1.8531241148182327

Density:0.05,type:tp,mae:53.93030007649933,mse:12220.139733058135,rmse:110.54474086567002
Density:0.1,type:tp,mae:53.84031790810964,mse:12205.200139834596,rmse:110.47714759095926
Density:0.15,type:tp,mae:53.69389621704939,mse:12233.362269357016,rmse:110.60453096214918
Density:0.2,type:tp,mae:53.95651903024888,mse:12114.249782573803,rmse:110.0647526802918
"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "tp"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    umean = UMEANModel()
    umean.fit(train_data)
    y, y_pred = umean.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

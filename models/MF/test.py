from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import MFModel
"""
T
RESULT MF:
Density:0.05,type:rt,mae:0.576378887814029,mse:2.082243794355493,rmse:1.4429981962412473
Density:0.1,type:rt,mae:0.49478550883456834,mse:1.6236234720660405,rmse:1.274214845332623
Density:0.15,type:rt,mae:0.46929335905591363,mse:1.4799293112956513,rmse:1.216523452834203
Density:0.2,type:rt,mae:0.4395753397852491,mse:1.370772157241582,rmse:1.1707997938339338

Density:0.05,type:tp,mae:25.81130047175078,mse:4872.765566549135,rmse:69.80519727462372
Density:0.1,type:tp,mae:20.673126816311118,mse:3994.3217322699775,rmse:63.20064661275213
Density:0.15,type:tp,mae:17.115162743452927,mse:3001.432680489639,rmse:54.78533271314175
"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "tp"
    latent_dim = 8
    lr = 0.0001
    lambda_ = 0.1
    epochs = 200
    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)

    mf = MFModel(md_data.row_n, md_data.col_n, latent_dim, lr, lambda_)
    mf.fit(train_data, test_data, epochs)
    y, y_pred = mf.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

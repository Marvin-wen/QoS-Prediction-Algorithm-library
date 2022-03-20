from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import MFModel
"""
RESULT MF:
Density:0.05,type:rt,mae:0.6716546313096282,mse:2.452621597109724,rmse:1.5660847988246753
Density:0.1,type:rt,mae:0.5169443118586329,mse:1.7229506095252545,rmse:1.3126121321720496
Density:0.15,type:rt,mae:0.47249373047649473,mse:1.5053596856557376,rmse:1.2269310028097495
Density:0.2,type:rt,mae:0.4395753397852491,mse:1.370772157241582,rmse:1.1707997938339338
"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "rt"
    latent_dim = 8
    lr = 0.001
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

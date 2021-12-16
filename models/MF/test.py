from data import MatrixDataset
from .model import MFModel
from utils.evaluation import mae,mse,rmse
from utils.model_util import freeze_random

"""
RESULT UPCC:

"""

freeze_random() # 冻结随机数 保证结果一致

for density in [0.05,0.1,0.15,0.2]:

    type_ = "rt"
    latent_dim = 16
    lr = 0.001
    lambda_ = 0.1
    epochs = 1000
    md_data = MatrixDataset(type_)
    train_data,test_data = md_data.split_train_test(density)


    mf = MFModel(md_data.row_n,md_data.col_n,latent_dim,lr,lambda_)
    mf.fit(train_data,test_data,epochs)
    y,y_pred = mf.predict(test_data)


    mae_ = mae(y,y_pred)
    mse_ = mse(y,y_pred)
    rmse_ = rmse(y,y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")



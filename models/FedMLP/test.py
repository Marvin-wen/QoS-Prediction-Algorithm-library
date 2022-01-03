import torch
from data import MatrixDataset, ToTorchDataset
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random
from root import absolute
from .model import FedMLPModel

"""
RESULT FedMLP:

"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "rt"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    lr = 0.001
    epochs = 100
    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.L1Loss()

    dim = 8

    mlp = FedMLPModel(train_data, loss_fn, rt_data.row_n, rt_data.col_n, dim=dim)
    opt = Adam(mlp.parameters(), lr=lr)

    mlp.fit(epochs,lr,test_dataloader)
    y, y_pred = mlp.predict(test_dataloader)
    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

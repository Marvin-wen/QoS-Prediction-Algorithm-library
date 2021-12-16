from torch import optim
from torch.nn.modules import loss
from data import MatrixDataset
from .model import MLPModel
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random
from data import ToTorchDataset


import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
"""
RESULT MLP:
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
    loss_fn = nn.SmoothL1Loss()

    dim = 8

    mlp = MLPModel(loss_fn, rt_data.row_n, rt_data.col_n, dim=12)
    opt = Adam(mlp.parameters(), lr=lr)

    mlp.fit(train_dataloader,epochs,opt,eval_loader=test_dataloader)
    # y, y_pred = mlp.predict(test_data, 20)

    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)

    # print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

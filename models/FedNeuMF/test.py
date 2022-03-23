import torch
from data import MatrixDataset, ToTorchDataset
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import FedNeuMFModel
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

    test_dataloader = DataLoader(test_dataset, batch_size=2048)

    lr = 0.0005
    epochs = 3000
    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.L1Loss()

    dim = 8

    neumf = FedNeuMFModel(
        train_data,
        loss_fn,
        rt_data.row_n,
        rt_data.col_n,
        dim=8,
        layers=[64, 32, 8],
    )

    neumf.fit(epochs, lr, test_dataloader)
    # y, y_pred = neumf.predict(
    #     test_dataloader, False,
    # )
    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)

    # print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

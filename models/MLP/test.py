import torch
from data import MatrixDataset, ToTorchDataset
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from root import absolute
from .model import MLPModel

# 冻结随机数
from utils.model_util import freeze_random
# 日志
from utils.mylogger import TNLog

"""
RESULT MLP:
Density:0.05,type:rt,mae:0.4674951136112213,mse:1.8543723821640015,rmse:1.3617534637451172

"""

freeze_random()  # 冻结随机数 保证结果一致

logger = TNLog('MLP')
logger.initial_logger()

for density in [0.05, 0.10, 0.15, 0.20]:
    type_ = "rt"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    lr = 0.01
    epochs = 100
    loss_fn = nn.L1Loss()
    dim = 12

    mlp = MLPModel(loss_fn, rt_data.row_n, rt_data.col_n, dim=dim)
    opt = Adam(mlp.parameters(), lr=lr)

    mlp.fit(train_dataloader, epochs, opt, eval_loader=test_dataloader, save_filename=f"Density-{density}")
    # y, y_pred = mlp.predict(test_dataloader, True,
    #                         "/Users/wenzhuo/Desktop/研究生/科研/QoS预测实验代码/SCDM/output/FedMLPModel/loss_0.4504.ckpt")
    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)
    #
    # logger.info(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

def predict():
    y, y_pred = mlp.predict(test_dataloader, True,
                            "/Users/wenzhuo/Desktop/研究生/科研/QoS预测实验代码/SCDM/output/FedMLPModel/loss_0.4504.ckpt")
    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    logger.info(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

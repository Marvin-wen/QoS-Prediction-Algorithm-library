import torch
from data import MatrixDataset, ToTorchDataset
from models.NeuMF.model import NeuMF, NeuMFModel
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
# 冻结随机数
from utils.model_util import freeze_random
# 日志
from utils.mylogger import TNLog
"""
RESULT NeuMF:
Density:0.05, type:rt, mae:0.6073, mse:3.6153, rmse:1.9014
Density:0.10, type:rt, mae:0.5820, mse:3.5959, rmse:1.8963
Density:0.15, type:rt, mae:0.5713, mse:3.5806, rmse:1.8922
Density:0.20, type:rt, mae:0.5651, mse:3.5951, rmse:1.8961
"""

freeze_random()  # 冻结随机数 保证结果一致

# logger = TNLog('NeuMF')
# logger.initial_logger()

path = [
    r'D:\SHUlpt\Code\QoS-Predcition-Algorithm-library\output\NeuMFModel\Density_0.05_loss-0.6073.ckpt',
    r'D:\SHUlpt\Code\QoS-Predcition-Algorithm-library\output\NeuMFModel\Density_0.1_loss-0.5820.ckpt',
    r'D:\SHUlpt\Code\QoS-Predcition-Algorithm-library\output\NeuMFModel\Density_0.15_loss-0.5713.ckpt',
    r'D:\SHUlpt\Code\QoS-Predcition-Algorithm-library\output\NeuMFModel\Density_0.2_loss-0.5651.ckpt'
]

for idx, density in enumerate([0.05, 0.10, 0.15, 0.20]):
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

    NeuMF = NeuMFModel(loss_fn, rt_data.row_n, rt_data.col_n, latent_dim=dim)
    opt = Adam(NeuMF.parameters(), lr=lr)

    NeuMF.fit(train_dataloader,
              epochs,
              opt,
              eval_loader=test_dataloader,
              save_filename=f"Density_{density}")

    # y, y_ = rmse(y, y_pred)

    # NeuMF.logger.info(
    #     f"Density:{density:.2f}, type:{type_}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}"
    # )

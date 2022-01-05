from collections import namedtuple
from functools import partial

import numpy as np
import torch
from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.FedXXX.model import Embedding, FedXXXLaunch, FedXXXModel
from models.FedXXX.resnet_utils import ResNetBasicBlock
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.decorator import timeit
from utils.evaluation import mae, mse, rmse
from utils.model_util import count_parameters, freeze_random

from .model import FedXXXModel
"""
RESULT MODEL:
"""

IS_FED = True

epochs = 1000
desnity = 0.05
type_ = "rt"

u_enable_columns = ["[User ID]", "[Country]"]
i_enable_columns = ["[Service ID]", "[Country]"]


def data_preprocess(traid,
                    u_info_obj: InfoDataset,
                    i_info_obj: InfoDataset,
                    is_dtraid=False):
    """生成d_traid [[traid],[p_traid]]
    """
    r = []
    for row in tqdm(traid, desc="Gen d_traid"):
        uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
        u = u_info_obj.query(uid)
        i = i_info_obj.query(iid)
        r.append([[uid, iid, rate], [u, i, rate]]) if is_dtraid else r.append(
            [u, i, rate])
    return r


fed_data_preprocess = partial(data_preprocess, is_dtraid=True)

md = MatrixDataset(type_)
u_info = InfoDataset("user", u_enable_columns)
i_info = InfoDataset("service", i_enable_columns)
train, test = md.split_train_test(desnity)

# loss_fn = nn.SmoothL1Loss()
loss_fn = nn.L1Loss()

user_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": u_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": [16, 16],
    "in_size": 32,  # embedding后接一个全连阶层在进入resnet
    "blocks_sizes": [64, 128, 64, 32],  # 最后的输出是8
    "deepths": [1, 2, 2],
    "activation": nn.GELU,
    "block": ResNetBasicBlock
}

item_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": [16, 16],
    "in_size": 32,
    "blocks_sizes": [64, 128, 64, 32],  # item最后的输出是8
    "deepths": [1, 2, 2],
    "activation": nn.GELU,
    "block": ResNetBasicBlock
}


def _check_params(params):
    if params["type_"] == "cat":
        embedding_dims = params["embedding_dims"]
        in_size = params["in_size"]
        assert sum(embedding_dims) == in_size
        deepths = params["deepths"]
        blocks_sizes = params["blocks_sizes"]
        assert len(blocks_sizes) - 1 == len(deepths)


_check_params(user_params)
_check_params(item_params)

if not IS_FED:
    train_data = data_preprocess(train, u_info, i_info)
    test_data = data_preprocess(test, u_info, i_info)
    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=2048)
    model = FedXXXModel(user_params, item_params, loss_fn, [64, 32, 16])  # 非联邦
    opt = Adam(model.parameters(), lr=0.0005)
    model.fit(train_dataloader, epochs, opt, eval_loader=test_dataloader)
    y, y_pred = model.predict(
        test_dataloader, True,
        "D:\yuwenzhuo\QoS-Predcition-Algorithm-library\output\FedXXXModel\loss_0.3477.ckpt"
    )
    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{0.05},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

else:
    train_data = fed_data_preprocess(train, u_info, i_info)
    test_data = fed_data_preprocess(test, u_info, i_info)
    model = FedXXXLaunch(train_data, user_params, item_params, [64, 32, 16],
                         loss_fn, 1, nn.GELU)

    print(f"模型参数:", count_parameters(model))
    model.fit(epochs, lr=0.0005, test_d_traid=test_data)
    # y, y_pred = model.predict(
    #     test_data,
    #     similarity_th=0.8,
    #     w=0.5,
    #     use_similarity=True,
    #     resume=True,
    #     path=
    #     "/Users/wenzhuo/Desktop/研究生/科研/QoS预测实验代码/SCDM/output/FedXXXLaunch/loss_0.4664.ckpt"
    # )
    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)
    # print(f"Density:{0.05},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

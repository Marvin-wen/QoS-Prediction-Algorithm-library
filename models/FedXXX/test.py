from collections import namedtuple
from functools import partial

import numpy as np
import torch
from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.FedXXX.model import Embedding, FedXXXLaunch, FedXXXModel
from models.FedXXX.resnet_utils import ResNetBasicBlock
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam, optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.decorator import timeit
from utils.evaluation import mae, mse, rmse
from utils.model_util import count_parameters, freeze_random

from .model import FedXXXModel
"""
RESULT MODEL:
Density:0.05,type:rt,mae:0.39733654260635376,mse:1.825060248374939,rmse:1.3509478569030762 43w
Density:0.1,type:rt,mae:0.40407416224479675,mse:1.8361570835113525,rmse:1.3550487756729126 44w
Density:0.15,type:rt,mae:0.3500382900238037,mse:1.5597128868103027,rmse:1.248884677886963 44
Density:0.2,type:rt,mae:0.34187352657318115,mse:1.5415867567062378,rmse:1.2416064739227295 43w

Density:0.05,type:tp,mae:17.988832473754883,mse:3172.656982421875,rmse:56.32634353637695
Density:0.1,type:tp,mae:15.86837100982666,mse:2433.6494140625,rmse:49.33203125
Density:0.15,type:tp,mae:14.093144416809082,mse:2032.4862060546875,rmse:45.08310317993164
Density:0.2,type:tp,mae:12.833450317382812,mse:1689.3419189453125,rmse:41.10160446166992


NON-FED
Density:0.05,type:rt,mae:0.3647790253162384,mse:1.7513489723205566,rmse:1.32338547706604
Density:0.1,type:rt,mae:0.3197784423828125,mse:1.4576799869537354,rmse:1.2073441743850708
Density:0.15,type:rt,mae:0.30833232402801514,mse:1.4249907732009888,rmse:1.1937297582626343
Density:0.2,type:rt,mae:0.28451427817344666,mse:1.2856311798095703,rmse:1.1338567733764648
"""

# non-fed 0.05 0.1 0.15 2
# fed 0.1 0.15 2

IS_FED = True

epochs = 3000
desnity = 0.05
type_ = "tp"

u_enable_columns = ["[User ID]", "[Country]", "[AS]"]
i_enable_columns = ["[Service ID]", "[Country]", "[AS]"]


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
    "embedding_dims": [16, 16, 16],
    "in_size": 48,  # embedding后接一个全连阶层在进入resnet
    "blocks_sizes": [64, 128, 64, 32],  # 最后的输出是8
    "deepths": [2, 2, 2],
    "activation": nn.GELU,
    "block": ResNetBasicBlock
}

item_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": [16, 16, 16],
    "in_size": 48,
    "blocks_sizes": [64, 128, 64, 32],  # item最后的输出是8
    "deepths": [2, 2, 2],
    "activation": nn.GELU,
    "block": ResNetBasicBlock
}

# user_params = {
#     "type_": "cat",  # embedding层整合方式 stack or cat
#     "embedding_nums": u_info.embedding_nums,  # 每个要embedding的特征的总个数
#     "embedding_dims": [8, 8,8],
#     "in_size": 24,  # embedding后接一个全连阶层在进入resnet
#     "blocks_sizes": [128,32, 12],  # 最后的输出是8
#     "deepths": [3,2],
#     "activation": nn.GELU,
#     "block": ResNetBasicBlock
# }

# item_params = {
#     "type_": "cat",  # embedding层整合方式 stack or cat
#     "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
#     "embedding_dims": [8, 8,8],
#     "in_size": 24,
#     "blocks_sizes": [128,32,12],  # item最后的输出是8
#     "deepths": [3,2],
#     "activation": nn.GELU,
#     "block": ResNetBasicBlock
# }


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
    model = FedXXXModel(user_params, item_params, loss_fn,
                        [64, 512, 128, 24])  # 非联邦
    opt = Adam(model.parameters(), lr=0.0005)
    print(f"模型参数:", count_parameters(model))
    model.fit(train_dataloader,
              epochs,
              opt,
              eval_loader=test_dataloader,
              save_filename=f"{desnity}_{type_}")
    # y, y_pred = model.predict(
    #     test_dataloader, True,
    #     "D:\yuwenzhuo\QoS-Predcition-Algorithm-library\output\FedXXXLaunch\loss_0.15_tp_10.3945.ckpt"
    # )
    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)

    # print(f"Density:{desnity},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

else:
    train_data = fed_data_preprocess(train, u_info, i_info)
    test_data = fed_data_preprocess(test, u_info, i_info)
    model = FedXXXLaunch(train_data,
                         user_params,
                         item_params, [64, 512, 128, 24],
                         loss_fn,
                         1,
                         nn.GELU,
                         optimizer="adam")

    print(f"模型参数:", count_parameters(model))
    # model.fit(epochs, lr=0.0005, test_d_traid=test_data, fraction=1,save_filename=f"{desnity}_{type_}")
    y, y_pred = model.predict(
        test_data,
        similarity_th=0.95,
        w=0.95,
        use_similarity=False,
        resume=True,
        path=
        "/Users/wenzhuo/Desktop/研究生/科研/QoS预测实验代码/SCDM/output/FedXXXLaunch/tp_0.05_44w_loss_12.9692.ckpt"
    )
    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)
    print(f"Density:{desnity},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

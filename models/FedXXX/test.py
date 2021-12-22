import numpy as np
import torch
from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.FedMF import client
from models.FedXXX.server import Server
from models.FedXXX.client import Clients
from models.FedXXX.model import Embedding, FedXXXModel,FedXXXLaunch
from models.FedXXX.utils import ResNetBasicBlock
from collections import namedtuple
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.decorator import timeit

from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from functools import partial

from .model import FedXXXModel

"""
RESULT MODEL:
"""

IS_FED = False

epochs = 1000
desnity = 0.05
type_ = "rt"

u_enable_columns = ["[User ID]", "[Country]"]
i_enable_columns = ["[Service ID]", "[Country]"]

def data_preprocess(traid,u_info_obj:InfoDataset,i_info_obj:InfoDataset,is_dtraid=False):
    """生成d_traid

    Args:
        traid ([type]): [description]
        u_info_obj (InfoDataset): [description]
        i_info_obj (InfoDataset): [description]
        need_uid (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    r = []
    for row in tqdm(traid,desc="Gen d_traid"):
        uid,iid,rate = int(row[0]),int(row[1]),float(row[2])
        u = u_info_obj.query(uid)
        i = i_info_obj.query(iid)
        r.append([[uid,iid,rate],[u,i,rate]]) if is_dtraid else r.append([u,i,rate])
    return r

fed_data_preprocess = partial(data_preprocess,is_dtraid=True)



md = MatrixDataset(type_)
u_info = InfoDataset("user",u_enable_columns)
i_info = InfoDataset("service",i_enable_columns)
train,test = md.split_train_test(desnity)

user_params = {
    "type_":"stack", # embedding层整合方式 stack or cat
    "embedding_nums":u_info.embedding_nums,# 每个要embedding的特征的总个数
    "embedding_dims":[8,8],
    "in_size":8, # embedding后接一个全连阶层在进入resnet
    "blocks_sizes":[8,8], # 最后的输出是8
    "deepths":[1],
    "activation":nn.ReLU,
    "block":ResNetBasicBlock
}

item_params = {
    "type_":"stack", # embedding层整合方式 stack or cat
    "embedding_nums":i_info.embedding_nums,# 每个要embedding的特征的总个数
    "embedding_dims":[8,8],
    "in_size":8,
    "blocks_sizes":[8], # item最后的输出是8
    "deepths":[1],
    "activation":nn.ReLU,
    "block":ResNetBasicBlock
}

loss_fn = nn.SmoothL1Loss()

if not IS_FED:
    train_data = data_preprocess(train,u_info,i_info)
    test_data = data_preprocess(test,u_info,i_info)
    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset,batch_size=128)
    model = FedXXXModel(user_params,item_params,loss_fn,[16]) # 非联邦
    opt = Adam(model.parameters(), lr=0.001)
    model.fit(train_dataloader,epochs,opt,eval_loader=test_dataloader)
    # y, y_pred = model.predict(test_dataloader,True,"D:\yuwenzhuo\QoS-Predcition-Algorithm-library\output\FedXXXModel\loss_0.3477.ckpt")
    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)

    print(f"Density:{0.05},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")


else:
    train_data = fed_data_preprocess(train,u_info,i_info)
    test_data = fed_data_preprocess(test,u_info,i_info)
    model = FedXXXLaunch(train_data,user_params,item_params,[16],loss_fn,1,nn.ReLU)
    model.fit(epochs,lr=0.001,test_d_traid=test_data)


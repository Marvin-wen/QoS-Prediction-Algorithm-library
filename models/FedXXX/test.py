import numpy as np
import torch
from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.FedXXX.model import Embedding, FedXXXModel
from models.FedXXX.utils import ResNetBasicBlock
from tqdm import tqdm
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import FedXXXModel

"""
RESULT MODEL:
"""

epochs = 1000
desnity = 0.05
type_ = "rt"
u_enable_columns = ["[User ID]", "[Country]"]
i_enable_columns = ["[Service ID]", "[Country]"]

def data_preprocess(traid,u_info_obj:InfoDataset,i_info_obj:InfoDataset):
    r = []
    for row in tqdm(traid):
        uid,iid,rate = int(row[0]),int(row[1]),float(row[2])
        u = u_info_obj.query(uid)
        i = i_info_obj.query(iid)
        r.append([u,i,rate])
    return r

md = MatrixDataset(type_)
u_info = InfoDataset("user",u_enable_columns)
i_info = InfoDataset("service",i_enable_columns)
train,test = md.split_train_test(desnity)
train_data = data_preprocess(train,u_info,i_info)
test_data = data_preprocess(test,u_info,i_info)
train_data = ToTorchDataset(train_data)
test_data = ToTorchDataset(test_data)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

user_params = {
    "type_":"stack", # embedding层整合方式 stack or cat
    "embedding_nums":u_info.embedding_nums,# 每个要embedding的特征的总个数
    "embedding_dims":[16,16],
    "in_size":16, # embedding后接一个全连阶层在进入resnet
    "blocks_sizes":[16,8], # 最后的输出是8
    "deepths":[2],
    "activation":nn.ReLU,
    "block":ResNetBasicBlock
}

item_params = {
    "type_":"stack", # embedding层整合方式 stack or cat
    "embedding_nums":i_info.embedding_nums,# 每个要embedding的特征的总个数
    "embedding_dims":[16,16],
    "in_size":16,
    "blocks_sizes":[16,8], # item最后的输出是8
    "deepths":[2],
    "activation":nn.ReLU,
    "block":ResNetBasicBlock
}

loss_fn = nn.SmoothL1Loss()
model = FedXXXModel(user_params,item_params,loss_fn,[16])

opt = Adam(model.parameters(), lr=0.001)
model.fit(train_dataloader,epochs,opt,eval_loader=test_dataloader)

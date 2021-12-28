import copy
import os
import random
import shutil

import numpy as np
import torch
from root import absolute
from scipy.sparse.construct import rand

"""
    Some handy functions for model training ...
"""

def save_checkpoint(state, is_best, save_dirname="output", save_filename="best_model.ckpt"):
    """Save checkpoint if a new best is achieved"""
    if not os.path.isdir(absolute(save_dirname)):
        os.makedirs(absolute(save_dirname))
    file_path= absolute(f"{save_dirname}/{save_filename}")
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, file_path)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")


def load_checkpoint(file_path: str, device=None):
    """Loads torch model from checkpoint file.
    Args:
        file_path (str): Path to checkpoint directory or filename
    """
    if not os.path.exists(file_path):
        raise Exception("ckpt file doesn't exist")
    ckpt = torch.load(file_path,map_location=device)
    print(' [*] Loading checkpoint from %s succeed!' % file_path)
    return ckpt




def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def freeze_random(seed=2021):
    
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    

def traid_to_matrix(traid,nan_symbol=-1):
    """三元组转矩阵

    Args:
        traid ([type]): [description]
        nan_symbol (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    # 注意下标应该为int
    if not isinstance(traid,np.ndarray):
        traid = np.array(traid)
    x_max = traid[:,0].max().astype(int)
    y_max = traid[:,1].max().astype(int)
    matrix = np.full((x_max+1,y_max+1),nan_symbol,dtype=traid.dtype)
    matrix[traid[:,0].astype(int),traid[:,1].astype(int)] = traid[:,2]
    return matrix

def split_d_traid(d_traid):
    l = np.array(d_traid,dtype=np.object)
    return np.array(l[:,0].tolist()), l[:,1].tolist()

def nonzero_mean(matrix,nan_symbol):
    m = copy.deepcopy(matrix)
    m[matrix==nan_symbol] = 0
    t = (m != 0).sum(axis=-1)
    return (m.sum(axis=-1) / t).squeeze()


def use_optimizer(network, opt):
    if opt == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=0.001,
                                    momentum=0.99
                                )
    elif opt == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                    lr=0.001,
                                    weight_decay=0.0005)
    return optimizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    d_traid = [
        [[1,2,3.2],[[1,1],[2,2],3.2]],
        [[1,2,3.2],[[1,1],[2,2],3.2]],
        [[1,2,3.2],[[1,1],[2,2],3.2]],
        [[1,2,3.2],[[1,1],[2,2],3.2]]

    ]
    a,b = split_d_traid(d_traid)
    t2m = traid_to_matrix(a)
    print(t2m)
    print(nonzero_mean(t2m,-1))

from scipy.sparse.construct import rand
import torch
import random
import numpy as np
import copy
"""
    Some handy functions for model training ...
"""

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
    x_max = traid[:,0].max().astype(int)
    y_max = traid[:,1].max().astype(int)
    matrix = np.full((x_max+1,y_max+1),nan_symbol,dtype=traid.dtype)
    matrix[traid[:,0].astype(int),traid[:,1].astype(int)] = traid[:,2]
    return matrix

def nonzero_mean(matrix,nan_symbol):
    m = copy.deepcopy(matrix)
    m[matrix==nan_symbol] = 0
    t = (m != 0).sum(axis=-1)
    return (m.sum(axis=-1) / t).squeeze()
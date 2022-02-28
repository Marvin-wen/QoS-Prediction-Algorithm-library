from re import X

import numpy as np
from sklearn import preprocessing
"""For More Information Read : https://ssjcoding.github.io/2019/03/27/normalization-and-standardization/
"""


def min_max_scaler(data: np.ndarray, scaler=None):
    """返回转换后的数据以及应用的转换器
    """
    if scaler:
        x = scaler.transform(data)
    else:
        scaler = preprocessing.MinMaxScaler()
        x = scaler.fit_transform(data)
    return x, scaler


def l2_norm(data: np.ndarray, scaler=None):
    if scaler:
        x = scaler.transform(data)
    else:
        scaler = preprocessing.Normalizer().fit(data)
        x = scaler.transform(data)
    return x, scaler


def z_score(data: np.ndarray, scaler=None):
    if scaler:
        x = scaler.transform(data)
    else:
        scaler = preprocessing.StandardScaler().fit(data)
        x = scaler.transform(data)
    return x, scaler

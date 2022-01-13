import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y, y_pred):
    return mean_absolute_error(y, y_pred)


def mse(y, y_pred):
    return mean_squared_error(y, y_pred)


def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))

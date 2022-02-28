import numpy as np
import math
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


# def euclidean_similarity(x, y):
#     """欧氏距离
#
#     Args:
#         x ():
#         y ():
#
#     Returns:
#
#     """
#     n = len(x)
#     if n != len(y):
#         raise ValueError('x and y must have the same length.')
#
#     x = np.asarray(x)
#     y = np.asarray(y)
#
#     dist = 0
#     for a, b in zip(x, y):
#         dist += (a - b) ** 2
#     return dist ** 0.5


def pearsonr_similarity(x, y):
    """皮尔森相似度

    Args:
        x ():
        y ():

    Returns:

    """
    return pearsonr(x, y)[0]


def cosine_similarity(x, y):
    """余弦相似度

    Args:
        x ():
        y ():

    Returns:

    """
    return cosine_similarity(x, y)


# FIXME 测试BUG
# TODO 写法优化
def adjusted_cosine_similarity(x, y):
    """修正的余弦相似度

    Args:
        x ():
        y ():

    Returns:

    """
    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')

    if n < 2:
        raise ValueError('x and y must have length at least 2.')

    x = np.asarray(x)
    y = np.asarray(y)

    # 求两个向量的共同交集
    nonzero_x = np.nonzero(x)[0]
    nonzero_y = np.nonzero(y)[0]
    intersect = np.intersect1d(nonzero_x, nonzero_y)

    if len(nonzero_x) < 2 or len(nonzero_y) < 2:
        raise ValueError('there must be at least two non-zero entries')

    mean_x = x.sum() / len(nonzero_x)
    mean_y = y.sum() / len(nonzero_y)

    multiply_sum = sum((x[i] - mean_x) * (y[i] - mean_y) for i in intersect)
    pow_sum_x = sum(math.pow(x[i] - mean_x, 2) for i in nonzero_x)
    pow_sum_y = sum(math.pow(y[i] - mean_y, 2) for i in nonzero_y)

    return multiply_sum / math.sqrt(pow_sum_x * pow_sum_y)

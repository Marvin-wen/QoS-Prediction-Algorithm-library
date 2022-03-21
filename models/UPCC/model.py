import copy
import numpy as np
import math
from models import UMEAN
from numpy.core.defchararray import expandtabs
from numpy.core.fromnumeric import nonzero
from tqdm import tqdm
from utils.model_util import nonzero_user_mean, triad_to_matrix

# 相似度计算库
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class UPCCModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = None  # QoS矩阵
        self.u_mean = None  # 每个用户的评分均值
        self.similarity_matrix = None  # 用户相似度矩阵
        self._nan_symbol = -1  # 缺失项标记（数据集中使用-1表示缺失项）

    def _get_similarity_matrix(self, matrix, metric):
        """获取项目相似度矩阵

        Args:
            matrix (): QoS矩阵
            metric (): 相似度计算方法, 可选参数: PCC(皮尔逊相关系数), COS(余弦相似度), ACOS(修正的余弦相似度)

        """
        _m = copy.deepcopy(matrix)
        _m[_m == self._nan_symbol] = 0  # 将缺失项用0代替，以便之后计算
        n_users = matrix.shape[0]
        similarity_matrix = np.zeros((n_users, n_users))

        # 计算相似度矩阵
        for i in tqdm(range(n_users), desc="生成相似度矩阵"):
            row_i = _m[i]
            nonzero_i = np.nonzero(row_i)[0]  # 非0元素所对应的下标
            for j in range(i + 1, n_users):
                row_j = _m[j]
                nonzero_j = np.nonzero(row_j)[0]
                intersect = np.intersect1d(nonzero_i,
                                           nonzero_j)  # 用户i,j共同评分过的项目交集
                if len(intersect) == 0:
                    sim = 0
                else:
                    # 依据指定的相似度计算方法计算用户i,j的相似度
                    try:
                        if metric == 'PCC':
                            # 如果一个项目的评分向量中所有值都相等，则无法计算皮尔逊相关系数
                            if len(set(row_i[intersect])) == 1 or len(
                                    set(row_j[intersect])) == 1:
                                sim = 0
                            else:
                                sim = pearsonr(row_i[intersect],
                                               row_j[intersect])[0]
                        elif metric == 'COS':
                            sim = cosine_similarity(row_i[intersect],
                                                    row_j[intersect])
                        elif metric == 'ACOS':
                            sim = adjusted_cosine_similarity(
                                row_i, row_j, intersect, i, j, self.u_mean)
                    except Exception as e:
                        sim = 0
                similarity_matrix[i][j] = similarity_matrix[j][i] = sim

        return similarity_matrix

    def _get_similarity_users(self, uid, topk=-1):
        """获取相似用户

        Args:
            uid (): 当前用户
            topk (): 相似用户数量, -1表示不限制数量

        Returns:
            依照相似度从大到小排序, 与当前用户最为相似的前topk个相似用户

        """
        assert isinstance(topk, int)
        ordered_sim_uid = (
            -self.similarity_matrix[uid]).argsort()  # 按相似度从大到小排序后, 相似用户对应的索引
        if topk == -1:
            return ordered_sim_uid
        else:
            assert topk > 0
            return ordered_sim_uid[:topk]

    def get_similarity(self, uid_a, uid_b):
        """传入两个uid，获取这两个用户的相似度
        """
        if uid_a == uid_b:
            return float(1)
        if uid_a + 1 > self.matrix.shape[0] or uid_b + 1 > self.matrix.shape[0]:
            return 0
        if self.similarity_matrix is None:
            assert self.matrix is not None, "Please fit first e.g. model.fit()"
            self._get_similarity_matrix(self.matrix)

        return self.similarity_matrix[uid_a][uid_b]

    def fit(self, triad, metric='PCC'):
        """训练模型

        Args:
            triad (): 数据三元组: (uid, iid, rating)
            metric (): 相似度计算方法, 可选参数: PCC(皮尔逊相关系数), COS(余弦相似度), ACOS(修正的余弦相似度)
        """
        self.matrix = triad_to_matrix(triad, self._nan_symbol)  # 数据三元组转QoS矩阵
        self.u_mean = nonzero_user_mean(self.matrix,
                                        self._nan_symbol)  # 根据QoS矩阵计算每个用户的评分均值
        self.similarity_matrix = self._get_similarity_matrix(
            self.matrix, metric)  # 根据QoS矩阵获取用户相似矩阵

    def predict(self, triad, topK=-1):
        y_list = []  # 真实评分
        y_pred_list = []  # 预测评分
        cold_boot_cnt = 0  # 冷启动统计
        assert self.u_mean is not None, "Please fit first e.g. model.fit()"

        for row in tqdm(triad, desc="Predict... "):
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            # 冷启动: 新用户因为没有计算过相似用户, 因此无法预测评分
            if uid + 1 > len(self.u_mean):
                cold_boot_cnt += 1
                continue
            u_mean = self.u_mean[uid]  # 当前用户评分均值
            similarity_users = self._get_similarity_users(uid, topK)
            up = 0  # 分子
            down = 0  # 分母
            # 对于当前用户的每一个相似用户
            for sim_uid in similarity_users:
                sim_user_rate = self.matrix[sim_uid][iid]  # 相似用户对目标item的评分
                similarity = self.get_similarity(uid, sim_uid)
                # 如果相似用户对目标item没有评分，或者相似度为负，则不进行计算
                if sim_user_rate == self._nan_symbol or similarity <= 0:
                    continue
                up += similarity * (sim_user_rate - self.u_mean[sim_uid]
                                    )  # 相似度 * (相似用户评分 - 相似用户评分均值)
                down += similarity

            if down != 0:
                y_pred = u_mean + up / down
            else:
                y_pred = u_mean

            y_pred_list.append(y_pred)
            y_list.append(rate)

        print(f"cold boot :{cold_boot_cnt / len(triad) * 100:4f}%")
        return y_list, y_pred_list


def adjusted_cosine_similarity(x, y, intersect, id_x, id_y, u_mean):
    """修正的余弦相似度

    Returns:

    """
    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    if n < 2:
        raise ValueError('x and y must have length at least 2.')
    if len(intersect) < 2:
        raise ValueError('there must be at least two non-zero entries')

    x = np.asarray(x)
    y = np.asarray(y)
    nonzero_x = np.nonzero(x)[0]
    nonzero_y = np.nonzero(y)[0]

    multiply_sum = sum(
        (x[i] - u_mean[id_x]) * (y[i] - u_mean[id_y]) for i in intersect)
    pow_sum_x = sum(math.pow(x[i] - u_mean[id_x], 2) for i in nonzero_x)
    pow_sum_y = sum(math.pow(y[i] - u_mean[id_y], 2) for i in nonzero_y)

    return multiply_sum / math.sqrt(pow_sum_x * pow_sum_y)


if __name__ == "__main__":
    triad = np.array([
        [0, 0, 1],
        [0, 1, 3],
        [1, 0, 1],
        [1, 1, 3],
        [1, 2, 4],
        [2, 0, 2],
        [2, 1, 3],
        [2, 2, 5],
    ])

    test = np.array([[0, 2, 3]])
    # y 3
    # y_pred 3.5

    upcc = UPCCModel()
    upcc.fit(triad)
    upcc.predict(test)

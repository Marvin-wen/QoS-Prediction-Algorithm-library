import copy
import math
import numpy as np
from tqdm import tqdm
from utils.model_util import traid_to_matrix, nonzero_user_mean, nonzero_item_mean

# 相似度计算库
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class IPCCModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = None  # QoS矩阵
        self.u_mean = None  # 每个用户的评分均值（用于计算修正的余弦相似度）
        self.i_mean = None  # 每个项目的评分均值
        self.similarity_matrix = None  # 项目相似度矩阵
        self._nan_symbol = -1  # 缺失项标记（数据集中使用-1表示缺失项）

    def _get_similarity_matrix(self, matrix, metric):
        """获取项目相似度矩阵

        Args:
            matrix (): QoS矩阵
            metric (): 相似度计算方法, 可选参数: PCC(皮尔逊相关系数), COS(余弦相似度), ACOS(修正的余弦相似度)

        """
        _m = copy.deepcopy(matrix)
        _m[_m == self._nan_symbol] = 0  # 将缺失项用0代替，以便之后计算
        n_items = matrix.shape[1]
        similarity_matrix = np.zeros((n_items, n_items))

        # 计算相似度矩阵
        for i in tqdm(range(n_items), desc="生成相似度矩阵"):
            for j in range(i + 1, n_items):
                col_i = _m[:, i]
                col_j = _m[:, j]
                nonzero_i = np.nonzero(col_i)[0]  # 非0元素对应的下标
                nonzero_j = np.nonzero(col_j)[0]
                intersect = np.intersect1d(nonzero_i, nonzero_j)  # 对项目i,j同时有评分的用户集合

                if len(intersect) == 0:
                    sim = 0
                else:
                    # 依据指定的相似度计算方法计算项目i,j的相似度
                    try:
                        if metric == 'PCC':
                            # 如果一个项目的评分向量中所有值都相等，则无法计算皮尔逊相关系数
                            if len(set(col_i[intersect])) == 1 or len(set(col_j[intersect])) == 1:
                                sim = 0
                            else:
                                sim = pearsonr(col_i[intersect], col_j[intersect])[0]
                        elif metric == 'COS':
                            sim = cosine_similarity(col_i[intersect], col_j[intersect])
                        elif metric == 'ACOS':
                            sim = adjusted_cosine_similarity(col_i, col_j, intersect, self.u_mean)
                    except Exception as e:
                        sim = 0

                similarity_matrix[i][j] = similarity_matrix[j][i] = sim

        return similarity_matrix

    def _get_similarity_items(self, iid, topk=-1):
        """获取相似用户

        Args:
            iid (): 当前项目
            topk (): 相似项目数量, -1表示不限制数量

        Returns:
            依照相似度从大到小排序, 与当前项目最为相似的前topk个相似项目

        """
        assert isinstance(topk, int)
        ordered_sim_iid = (-self.similarity_matrix[iid]).argsort()  # 按相似度从大到小排序后, 相似用户对应的索引
        if topk == -1:
            return ordered_sim_iid
        else:
            assert topk > 0
            return ordered_sim_iid[:topk]

    def get_similarity(self, iid_a, iid_b):
        """传入两个uid，获取这两个用户的相似度
        """
        if iid_a == iid_b:
            return float(1)
        if iid_a + 1 > self.matrix.shape[1] or iid_b + 1 > self.matrix.shape[1]:
            return 0
        if self.similarity_matrix is None:
            assert self.matrix is not None, "Please fit first e.g. model.fit()"
            self._get_similarity_matrix(self.matrix)

        return self.similarity_matrix[iid_a][iid_b]

    def fit(self, traid, metric='PCC'):
        """训练模型

        Args:
            traid (): 数据三元组: (uid, iid, rating)
            metric (): 相似度计算方法, 可选参数: PCC(皮尔逊相关系数), COS(余弦相似度), ACOS(修正的余弦相似度)
        """
        self.matrix = traid_to_matrix(traid, self._nan_symbol)  # 数据三元组转QoS矩阵
        self.u_mean = nonzero_user_mean(self.matrix, self._nan_symbol)  # 根据QoS矩阵计算每个用户的评分均值
        # FIXME 考虑i_mean为0的情况
        self.i_mean = nonzero_item_mean(self.matrix, self._nan_symbol)  # 根据QoS矩阵计算每个项目的评分均值
        self.similarity_matrix = self._get_similarity_matrix(self.matrix, metric)  # 根据QoS矩阵获取项目相似矩阵

    def predict(self, traid, topK=-1):
        y_list = []  # 真实评分
        y_pred_list = []  # 预测评分
        cold_boot_cnt = 0  # 冷启动统计

        for row in tqdm(traid, desc="Predict... "):
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            # 冷启动: 新用户因为没有计算过相似用户, 因此无法预测评分
            if iid + 1 > self.matrix.shape[1]:
                cold_boot_cnt += 1
                continue
            i_mean = self.i_mean[iid]
            similarity_items = self._get_similarity_items(iid, topK)
            up = 0  # 分子
            down = 0  # 分母
            # 对于当前项目的每一个相似项目
            for sim_iid in similarity_items:
                sim_item_rate = self.matrix[uid][sim_iid]  # 当前用户对相似项目的评分
                similarity = self.get_similarity(iid, sim_iid)
                # 如果当前用户对相似项目没有评分，则不进行计算
                if sim_item_rate == self._nan_symbol:
                    continue
                up += similarity * (sim_item_rate - self.i_mean[sim_iid])  # 相似度 * (相似项目评分 - 相似项目评分均值)
                down += similarity  # 相似度的绝对值

            if down != 0:
                y_pred = i_mean + up / down
            else:
                y_pred = 0

            y_pred_list.append(y_pred)
            y_list.append(rate)

        print(f"cold boot :{cold_boot_cnt / len(traid) * 100:4f}%")
        return y_list, y_pred_list


def adjusted_cosine_similarity(x, y, intersect, u_mean):
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

    multiply_sum = sum((x[i] - u_mean[i]) * (y[i] - u_mean[i]) for i in intersect)
    pow_sum_x = sum(math.pow(x[i] - u_mean[i], 2) for i in intersect)
    pow_sum_y = sum(math.pow(y[i] - u_mean[i], 2) for i in intersect)

    return multiply_sum / math.sqrt(pow_sum_x * pow_sum_y)


if __name__ == "__main__":
    traid = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 3],
        [1, 2, 4],
        [2, 0, 2],
        [2, 1, 3],
        [2, 2, 5],
    ])

    test = np.array([[0, 2, 3]])

    ipcc = IPCCModel()
    ipcc.fit(traid)
    ipcc.predict(test, 20)

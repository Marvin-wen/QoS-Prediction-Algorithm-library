import copy

import numpy as np
from models import UMEAN
from numpy.core.defchararray import expandtabs
from numpy.core.fromnumeric import nonzero
from scipy.stats import pearsonr
from tqdm import tqdm
from utils.model_util import nonzero_mean, triad_to_matrix


class UPCCModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = None
        self.u_mean = None
        self.similarity_matrix = None
        self._nan_symbol = -1

    def _get_similarity_matrix(self, matrix):
        """获取相似矩阵,暂时只支持PCC

        Args:
            matrix (): 由三元组转换得到的矩阵

        """
        _m = copy.deepcopy(matrix)
        _m[_m == self._nan_symbol] = 0

        n_users = matrix.shape[0]
        similarity_matrix = np.zeros((n_users, n_users))  # 任意两个用户之间的相似度
        # 这里复杂度还是有点高
        for i in tqdm(range(n_users), desc="生成相似度矩阵"):
            row_i = _m[i]
            nonzero_i = np.nonzero(row_i)[0]
            for j in range(i + 1, n_users):
                row_j = _m[j]
                nonzero_j = np.nonzero(row_j)[0]
                intersect = np.intersect1d(nonzero_i, nonzero_j)  # 两个用户的交集
                if len(intersect) == 0:
                    sim = 0
                else:
                    try:
                        sim = pearsonr(row_i[intersect], row_j[intersect])[0]
                    except Exception as e:
                        sim = 0
                similarity_matrix[i][j] = sim

        return similarity_matrix

    def _get_similarity_users(self, uid, topk=-1):
        assert isinstance(topk, int)
        r = (-self.similarity_matrix[uid]).argsort()
        if topk == -1:
            return r
        else:
            assert topk > 0
            return r[:topk]

    def get_similarity(self, uid_a, uid_b):
        """传入两个uid，获取这两个user的相似度
        """
        if uid_a == uid_b:
            return float(1)
        if uid_a + 1 > self.similarity_matrix.shape[0]:
            return 0
        if self.similarity_matrix is None:
            assert self.matrix is not None, "Please fit first e.g. model.fit()"
            self._get_similarity_matrix(self.matrix)
        return self.similarity_matrix[uid_a][uid_b]

    def fit(self, triad):
        self.matrix = triad_to_matrix(triad, self._nan_symbol)  # 三元组转矩阵
        self.similarity_matrix = self._get_similarity_matrix(
            self.matrix)  # 根据矩阵获取相似矩阵
        self.u_mean = nonzero_mean(self.matrix, self._nan_symbol)  # 算好均值

    def predict(self, triad, topK=-1):
        y_list = []
        y_pred_list = []
        cold_boot_cnt = 0
        assert self.u_mean is not None, "Please fit first e.g. model.fit()"

        for row in tqdm(triad, desc="Predict... "):
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            if uid + 1 > len(self.u_mean):
                cold_boot_cnt += 1  # 冷启动
                continue
            u_mean = self.u_mean[uid]
            similarity_users = self._get_similarity_users(uid, topK)
            up = 0
            down = 0
            # 对于当前用户的每一个相似用户
            for s_uid in similarity_users:
                s_r_ui = self.matrix[s_uid][iid]  # 相似用户对目标item访问的值
                sim = self.get_similarity(uid, s_uid)
                if s_r_ui == self._nan_symbol or sim <= 0:
                    continue
                diff = s_r_ui - self.u_mean[s_uid]
                up += sim * diff  # 分子
                down += sim  # 分母

            if down != 0:
                y_pred = u_mean + up / down
            else:
                y_pred = u_mean

            y_pred_list.append(y_pred)
            y_list.append(rate)
        print(f"cold boot :{cold_boot_cnt/len(triad)*100:4f}%")
        return y_list, y_pred_list


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
    upcc.train(test)

import copy
import numpy as np
import math
from models import UMEAN
from numpy.core.defchararray import expandtabs
from numpy.core.fromnumeric import nonzero
from tqdm import tqdm
from utils.model_util import nonzero_user_mean, traid_to_matrix, nonzero_item_mean

# 相似度计算库
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class UIPCCModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = None  # QoS矩阵
        self.u_mean = None  # 每个用户的QoS均值
        self.i_mean = None  # 每个项目的QoS均值
        self.similarity_user_matrix = None  # 用户相似度矩阵
        self.similarity_item_matrix = None  # 项目相似度矩阵
        self._nan_symbol = -1  # 缺失项标记（数据集中使用-1表示缺失项）

    def _get_similarity_matrix(self):
        """获取用户相似度矩阵和项目相似度矩阵
        """
        matrix = copy.deepcopy(self.matrix)
        matrix[matrix == self._nan_symbol] = 0  # 将缺失项用0代替，以便之后计算
        m, n = matrix.shape
        similarity_user_matrix = np.zeros((m, m))
        similarity_item_matrix = np.zeros((n, n))

        # 计算用户相似度矩阵
        for i in tqdm(range(m), desc="生成用户相似度矩阵"):
            for j in range(i + 1, m):
                row_i = matrix[i]
                row_j = matrix[j]
                nonzero_i = np.nonzero(row_i)[0]
                nonzero_j = np.nonzero(row_j)[0]
                intersect = np.intersect1d(nonzero_i, nonzero_j)  # 用户i,j共同调用过的项目集合
                # 如果用户i,j共同调用过的项目集合为空, 则相似度为0
                # 如果用户i或者j的QoS值向量都相等, 则皮尔逊相关系数无法计算(分母为0)
                if len(intersect) == 0 or \
                        len(set(row_i[intersect])) == 1 or \
                        len(set(row_j[intersect])) == 1:
                    sim = 0
                else:
                    try:
                        sim = pearsonr(row_i[intersect], row_j[intersect])[0]
                        sim = (2 * len(intersect) / (len(nonzero_j) + len(nonzero_j))) * sim  # 使用增强的PCC计算公式
                    except Exception as e:
                        sim = 0
                similarity_user_matrix[i][j] = similarity_user_matrix[j][i] = sim

        # 计算项目相似度矩阵
        for i in tqdm(range(n), desc="生成项目相似度矩阵"):
            for j in range(i + 1, n):
                col_i = matrix[:, i]
                col_j = matrix[:, j]
                nonzero_i = np.nonzero(col_i)[0]
                nonzero_j = np.nonzero(col_j)[0]
                intersect = np.intersect1d(nonzero_i, nonzero_j)  # 同时调用过项目i,j的用户集合
                # 如果同时调用过项目i,j的用户集合为空, 则相似度为0
                # 如果项目i或者j的QoS值向量都相等, 则皮尔逊相关系数无法计算(分母为0)
                if len(intersect) == 0 or \
                        len(set(col_i[intersect])) == 1 or \
                        len(set(col_j[intersect])) == 1:
                    sim = 0
                else:
                    try:
                        sim = pearsonr(col_i[intersect], col_j[intersect])[0]
                        sim = (2 * len(intersect) / (len(nonzero_j) + len(nonzero_j))) * sim  # 使用增强的PCC计算公式
                    except Exception as e:
                        sim = 0
                similarity_item_matrix[i][j] = similarity_item_matrix[j][i] = sim

        return similarity_user_matrix, similarity_item_matrix

    def _get_similarity_users(self, uid, topk=-1):
        """获取topk个相似用户

        Args:
            uid (): 当前用户id
            topk (): 相似用户数量, -1表示不限制数量

        Returns:
            依照相似度从大到小排序, 与当前用户最为相似的前topk个相似用户

        """
        assert isinstance(topk, int)
        ordered_sim_uid = (-self.similarity_user_matrix[uid]).argsort()  # 按相似度从大到小排序后, 相似用户对应的索引
        ordered_sim_uid = [sim_uid for sim_uid in ordered_sim_uid if
                           self.similarity_user_matrix[sim_uid][uid] > 0]  # 只考虑相似度大于0的相似用户
        if topk == -1:
            return ordered_sim_uid
        else:
            assert topk > 0
            return ordered_sim_uid[:topk]

    def _get_similarity_items(self, iid, topk=-1):
        """获取topk个相似项目

        Args:
            uid (): 当前项目id
            topk (): 相似项目数量, -1表示不限制数量

        Returns:
            依照相似度从大到小排序, 与当前项目最为相似的前topk个相似项目

        """
        assert isinstance(topk, int)
        ordered_sim_iid = (-self.similarity_item_matrix[iid]).argsort()  # 按相似度从大到小排序后, 相似项目对应的索引
        ordered_sim_iid = [sim_iid for sim_iid in ordered_sim_iid if
                           self.similarity_item_matrix[sim_iid][iid] > 0]  # 只考虑相似度大于0的相似用户
        if topk == -1:
            return ordered_sim_iid
        else:
            assert topk > 0
            return ordered_sim_iid[:topk]

    def get_user_similarity(self, uid_a, uid_b):
        """传入两个用户的id，获取这两个用户的相似度
        """
        if uid_a == uid_b:
            return float(1)
        if uid_a + 1 > self.matrix.shape[0] or uid_b + 1 > self.matrix.shape[0]:
            return 0
        return self.similarity_user_matrix[uid_a][uid_b]

    def get_item_similarity(self, iid_a, iid_b):
        """传入两个uid，获取这两个用户的相似度
        """
        if iid_a == iid_b:
            return float(1)
        if iid_a + 1 > self.matrix.shape[1] or iid_b + 1 > self.matrix.shape[1]:
            return 0
        return self.similarity_item_matrix[iid_a][iid_b]

    def _upcc(self, uid, iid, similarity_users, u_mean):
        up = 0
        down = 0
        for sim_uid in similarity_users:  # 对于目标用户的每一个相似用户
            sim_user_rate = self.matrix[sim_uid][iid]  # 相似用户对目标项目的评分
            similarity = self.get_user_similarity(sim_uid, uid)  # 相似用户与目标用户的相似度
            up += similarity * (sim_user_rate - self.u_mean[sim_uid])
            down += similarity
        if down != 0:
            y_pred = u_mean + up / down
        else:
            y_pred = u_mean
        return y_pred

    def _ipcc(self, uid, iid, similarity_items, i_mean):
        up = 0
        down = 0
        for sim_iid in similarity_items:  # 对于目标用户的每一个相似用户
            sim_item_rate = self.matrix[uid][sim_iid]  # 相似用户对目标项目的评分
            similarity = self.get_user_similarity(sim_iid, uid)  # 相似用户与目标用户的相似度
            up += similarity * (sim_item_rate - self.i_mean[sim_iid])
            down += similarity
        if down != 0:
            y_pred = i_mean + up / down
        else:
            y_pred = i_mean
        return y_pred

    def fit(self, traid):
        """训练模型

        Args:
            traid (): 数据三元组: (uid, iid, rating)
        """
        self.matrix = traid_to_matrix(traid, self._nan_symbol)  # 数据三元组转用户项目矩阵
        self.u_mean = nonzero_user_mean(self.matrix, self._nan_symbol)  # 根据用户项目矩阵计算每个用户调用项目的QoS均值
        self.i_mean = nonzero_item_mean(self.matrix, self._nan_symbol)  # 根据用户项目矩阵计算每个项目被用户调用的QoS均值
        self.similarity_user_matrix, self.similarity_item_matrix = self._get_similarity_matrix()  # 获取用户相似度矩阵和项目相似度矩阵

    def predict(self, traid, topK_u=-1, topK_i=-1, lamb=0.5):
        y_list = []  # 真实评分
        y_pred_list = []  # 预测评分
        cold_boot_cnt = 0  # 冷启动统计
        assert self.matrix is not None, "Please fit first e.g. model.fit()"

        for row in tqdm(traid, desc="Predict... "):
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            # 冷启动: 新用户因为没有计算过相似用户, 因此无法预测评分, 新项目同理
            if uid + 1 > self.matrix.shape[0] or iid + 1 > self.matrix.shape[1]:
                cold_boot_cnt += 1
                continue

            u_mean = self.u_mean[uid]
            i_mean = self.i_mean[iid]
            similarity_users = self._get_similarity_users(uid, topK_u)
            similarity_items = self._get_similarity_items(iid, topK_i)

            # 计算置信度
            con_u = 0  # 用户置信度(user confidence weight)
            con_i = 0  # 项目置信度(item confidence weight)
            similarity_users_sum = sum([self.similarity_user_matrix[sim_uid][uid] for sim_uid in similarity_users])
            similarity_items_sum = sum([self.similarity_item_matrix[sim_iid][iid] for sim_iid in similarity_items])
            for sim_uid in similarity_users:
                up = self.similarity_user_matrix[sim_uid][uid]
                down = similarity_users_sum
                con_u += (up / down) * self.similarity_user_matrix[sim_uid][uid]
            for sim_iid in similarity_items:
                up = self.similarity_item_matrix[sim_iid][iid]
                down = similarity_items_sum
                con_i += (up / down) * self.similarity_item_matrix[sim_iid][iid]
            w_u = 1.0 * (con_u * lamb) / (con_u * lamb + con_i * (1 - lamb))
            w_i = 1.0 - w_u

            if len(similarity_users) == 0 and len(similarity_items) == 0:  # 相似用户和相似项目都不存在
                y_pred = w_u * u_mean + w_i * i_mean
            elif len(similarity_items) == 0:  # 只存在相似用户
                y_pred = self._upcc(uid, iid, similarity_users, u_mean)
            elif len(similarity_users) != 0:  # 只存在相似服务
                y_pred = self._ipcc(uid, iid, similarity_items, i_mean)
            else:  # 相似用户和相似项目都存在
                y_pred = w_u * self._upcc(uid, iid, similarity_users, u_mean) + \
                         w_i * self._ipcc(uid, iid, similarity_items, i_mean)
            y_pred_list.append(y_pred)
            y_list.append(rate)

        print(f"cold boot :{cold_boot_cnt / len(traid) * 100:4f}%")
        return y_list, y_pred_list


if __name__ == "__main__":
    traid = np.array([
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

    uipcc = UIPCCModel()
    uipcc.fit(traid)
    uipcc.predict(test)

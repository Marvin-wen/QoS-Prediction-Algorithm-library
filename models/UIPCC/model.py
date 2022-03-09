import copy

import numpy as np
# 相似度计算库
from scipy.stats import pearsonr
from tqdm import tqdm
from utils.model_util import (nonzero_item_mean, nonzero_user_mean,
                              traid_to_matrix)


def cal_similarity_matrix(x, y):
    """计算两个向量的增强皮尔逊相关系数
    """
    nonzero_x = np.nonzero(x)[0]
    nonzero_y = np.nonzero(y)[0]
    intersect = np.intersect1d(nonzero_x, nonzero_y)  # 交集
    # 如果向量交集为空，则相似度为0
    # 如果一个向量中所有值都相等，则无法计算皮尔逊相关距离(分母为0)
    if len(intersect) == 0 or len(set(x[intersect])) == 1 or len(
            set(y[intersect])) == 1:
        sim = 0
    else:
        try:
            sim = pearsonr(x[intersect], y[intersect])[0]
            sim = (2 * len(intersect) /
                   (len(nonzero_x) + len(nonzero_y))) * sim  # 增强PCC
        except Exception as e:
            sim = 0
    return sim


def cal_topk(similarity_matrix, id, topk):
    assert isinstance(topk, int)
    ordered_id = (
        -similarity_matrix[id]).argsort()  # 按相似度从大到小排序后, 相似用户/项目对应的索引
    ordered_id = [
        sim_id for sim_id in ordered_id if similarity_matrix[sim_id][id] > 0
    ]  # 只考虑相似度大于0的相似用户/服务
    if topk == -1:
        return ordered_id
    else:
        assert topk > 0
        return ordered_id[:topk]


class UIPCCModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = None  # QoS矩阵
        self.u_mean = None  # 每个用户的QoS均值
        self.i_mean = None  # 每个项目的QoS均值
        self.similarity_user_matrix = None  # 用户相似度矩阵
        self.similarity_item_matrix = None  # 项目相似度矩阵
        self._nan_symbol = -1  # 缺失项标记（数据集中使用-1表示缺失项）

    def get_similarity_matrix(self):
        """获取用户相似度矩阵和项目相似度矩阵
        """
        matrix = copy.deepcopy(self.matrix)
        matrix[matrix == self._nan_symbol] = 0  # 将缺失项用0代替，以便之后计算
        m, n = matrix.shape
        similarity_user_matrix = np.zeros((m, m))
        similarity_item_matrix = np.zeros((n, n))

        # 计算用户相似度矩阵

        row_idx, col_idx = np.nonzero(matrix)
        for i in tqdm(range(m), desc="生成用户相似度矩阵"):
            for j in range(i + 1, m):
                nonzero_i = col_idx[row_idx == i]
                nonzero_j = col_idx[row_idx == j]
                row_i = matrix[i]
                row_j = matrix[j]
                similarity_user_matrix[i][j] = similarity_user_matrix[j][
                    i] = cal_similarity_matrix(row_i, row_j)

        # 计算项目相似度矩阵
        for i in tqdm(range(n), desc="生成项目相似度矩阵"):
            for j in range(i + 1, n):
                col_i = matrix[:, i]
                col_j = matrix[:, j]
                similarity_item_matrix[i][j] = similarity_item_matrix[j][
                    i] = cal_similarity_matrix(col_i, col_j)

        return similarity_user_matrix, similarity_item_matrix

    def get_similarity_users(self, uid, topk=-1):
        """获取前topk个相似用户
        """
        return cal_topk(self.similarity_user_matrix, uid, topk)

    def get_similarity_items(self, iid, topk=-1):
        """获取前topk个相似项目
        """
        return cal_topk(self.similarity_item_matrix, iid, topk)

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
            similarity = self.get_user_similarity(sim_uid,
                                                  uid)  # 相似用户与目标用户的相似度
            if sim_user_rate == self._nan_symbol:
                continue
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
        for sim_iid in similarity_items:  # 对于目标项目的每一个相似项目
            sim_item_rate = self.matrix[uid][sim_iid]  # 目标用户对相似项目的评分
            similarity = self.get_item_similarity(sim_iid,
                                                  iid)  # 相似项目与目标项目的相似度
            if sim_item_rate == self._nan_symbol:
                continue
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
        self.u_mean = nonzero_user_mean(
            self.matrix, self._nan_symbol)  # 根据用户项目矩阵计算每个用户调用项目的QoS均值
        self.i_mean = nonzero_item_mean(
            self.matrix, self._nan_symbol)  # 根据用户项目矩阵计算每个项目被用户调用的QoS均值
        self.similarity_user_matrix, self.similarity_item_matrix = self.get_similarity_matrix(
        )  # 获取用户相似度矩阵和项目相似度矩阵

    def predict(self, traid, topk_u=-1, topk_i=-1, lamb=0.5):
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
            similarity_users = self.get_similarity_users(uid, topk_u)
            similarity_items = self.get_similarity_items(iid, topk_i)

            # 计算置信度
            con_u = 0  # 用户置信度(user confidence weight)
            con_i = 0  # 项目置信度(item confidence weight)
            similarity_users_sum = sum([
                self.similarity_user_matrix[sim_uid][uid]
                for sim_uid in similarity_users
            ])
            similarity_items_sum = sum([
                self.similarity_item_matrix[sim_iid][iid]
                for sim_iid in similarity_items
            ])
            for sim_uid in similarity_users:
                up = self.similarity_user_matrix[sim_uid][uid]
                down = similarity_users_sum
                con_u += (up /
                          down) * self.similarity_user_matrix[sim_uid][uid]
            for sim_iid in similarity_items:
                up = self.similarity_item_matrix[sim_iid][iid]
                down = similarity_items_sum
                con_i += (up /
                          down) * self.similarity_item_matrix[sim_iid][iid]
            w_u = 1.0 * (con_u * lamb) / (con_u * lamb + con_i * (1.0 - lamb))
            w_i = 1.0 - w_u

            if len(similarity_users) == 0 and len(
                    similarity_items) == 0:  # 相似用户和相似项目都不存在
                y_pred = w_u * u_mean + w_i * i_mean
            elif len(similarity_items) == 0:  # 只存在相似用户
                y_pred = self._upcc(uid, iid, similarity_users, u_mean)
            elif len(similarity_users) == 0:  # 只存在相似服务
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

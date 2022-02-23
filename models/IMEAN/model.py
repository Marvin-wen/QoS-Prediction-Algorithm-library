from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


class IMEANModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.i_mean = {}

    def fit(self, triad):
        i_invoked = defaultdict(list)  # 用户调用服务的值字典
        print("Prepare imeans...")
        for row in tqdm(triad):
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            i_invoked[iid].append(rate)
        for iid, rate_lis in i_invoked.items():
            self.i_mean[iid] = np.average(rate_lis)
        print("Prepare imeans done!")

    def predict(self, triad, cold_boot=None):
        assert self.i_mean != {}, "Please fit first. e.g. model.fit(triad)"
        y_lis = []
        y_pred_lis = []
        cold_boot_cnt = 0
        print("Predicting...")
        for row in tqdm(triad):
            uid, iid, y = int(row[0]), int(row[1]), float(row[2])
            y_pred = self.i_mean.get(iid, cold_boot)
            if y_pred == None:
                cold_boot_cnt += 1
                continue
            y_lis.append(y)
            y_pred_lis.append(y_pred)
        print(
            f"Predicting done! cold_boot:{cold_boot_cnt/len(triad)*100:.4f}%")
        return y_lis, y_pred_lis

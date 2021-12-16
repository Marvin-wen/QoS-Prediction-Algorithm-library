import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class IMEANModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.i_mean = {}

    def fit(self,traid):
        i_invoked = defaultdict(list) # 用户调用服务的值字典
        print("Prepare imeans...")
        for row in tqdm(traid):
            uid,iid,rate = int(row[0]),int(row[1]),float(row[2])
            i_invoked[iid].append(rate)
        for iid,rate_lis in i_invoked.items():
            self.i_mean[iid] = np.average(rate_lis)
        print("Prepare imeans done!")

    def predict(self,traid,code_boot=None):
        assert self.i_mean != {},"Please fit first. e.g. model.fit(traid)"
        y_lis = []
        y_pred_lis = []
        code_boot_cnt = 0
        print("Predicting...")
        for row in tqdm(traid):
            uid,iid,y = int(row[0]),int(row[1]),float(row[2])
            y_pred = self.i_mean.get(iid,code_boot)
            if y_pred == None:
                code_boot_cnt += 1
                continue
            y_lis.append(y)
            y_pred_lis.append(y_pred)
        print(f"Predicting done! code_boot:{code_boot_cnt/len(traid)*100:.4f}%")
        return y_lis,y_pred_lis


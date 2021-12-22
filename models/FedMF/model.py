from .server import Server
from .client import Clients
from tqdm import tqdm

class FedMF(object):
    def __init__(self,server:Server,clients:Clients) -> None:
        super().__init__()
        self.server = server
        self.clients = clients
    
    def fit(self,epochs,lambda_,lr):
        for epoch in tqdm(range(epochs),desc="Epochs"):
            gradient_from_user = []
            # 遍历每一个用户
            for client_id,client in self.clients:
                # client upgrade
                # 用户的每一条调用记录
                for row in client.traid:
                    uid,iid,rate = int(row[0]),int(row[1]),float(row[2])
                    # 获得预测值
                    y_pred = client.user_vec @ self.server.items_vec[iid].T
                    e_ui = rate - y_pred
                    # 计算梯度
                    user_grad = -2 * e_ui * self.server.items_vec[iid] + 2 * lambda_ * client.user_vec
                    item_grad = -2 * e_ui * client.user_vec + 2 * lambda_ * self.server.items_vec[iid]
                    # 在用户端更新用户特征矩阵
                    client.user_vec -= lr * user_grad
                    # 收集梯度信息
                    gradient_from_user.append([iid,item_grad])
            # server upgrade
            # 在服务端更新服务特征矩阵
            self.server.upgrade(lr,gradient_from_user)
                    
    
    def predict(self,traid):
        y_list = []
        y_pred_list = []
        for row in traid:
            uid,iid,rate = int(row[0]),int(row[1]),float(row[2])
            y_pred = self.clients[uid].user_vec @ self.server.items_vec[iid].T
            y_list.append(rate)
            y_pred_list.append(y_pred)
        return y_list,y_pred_list    

from collections import OrderedDict
from typing import Dict, List

from tqdm import tqdm

from utils.model_util import use_optimizer

from .utils import train_mult_epochs_with_dataloader


class ClientBase(object):
    def __init__(self, device, model) -> None:
        self.device = device
        self.model = model
        self.data_loader = None
        super().__init__()

    def fit(self, params, loss_fn, optimizer: str, lr, epochs=5):
        self.model.load_state_dict(params)
        self.model.train()
        self.model.to(self.device)
        opt = use_optimizer(self.model, optimizer, lr)
        loss, lis = train_mult_epochs_with_dataloader(
            epochs,
            model=self.model,
            device=self.device,
            dataloader=self.data_loader,
            opt=opt,
            loss_fn=loss_fn)
        self.loss_list = [*lis]
        return self.model.state_dict(), round(loss, 4)


class ServerBase(object):
    def __init__(self) -> None:
        super().__init__()

    def upgrade_wich_cefficients(self, params: List[Dict], coefficients: Dict):
        """使用加权平均对参数进行更新

        Args:
            params : 模型参数
            coefficients : 加权平均的系数
        """

        o = OrderedDict()
        if len(params) != 0:
            # 获得不同的键
            for k, v in params[0].items():
                for it, param in enumerate(params):
                    if it == 0:
                        o[k] = coefficients[it] * param[k]
                    else:
                        o[k] += coefficients[it] * param[k]
            self.params = o

    def upgrade_average(self, params: List[Dict]):
        o = OrderedDict()
        if len(params) != 0:
            for k, v in params[0].items():
                o[k] = sum([i[k] for i in params]) / len(params)
            self.params = o


class FedModelBase(object):
    def update_selected_clients(self, sampled_client_indices, lr, s_params):
        """使用 client.fit 函数来训练被选择的client
        """
        collector = []
        client_loss = []
        selected_total_size = 0  # client数据集总数

        for uid in tqdm(sampled_client_indices, desc="Client training"):
            s_params, loss = self.clients[uid].fit(s_params, self.loss_fn,
                                                   self.optimizer, lr)
            collector.append(s_params)
            client_loss.append(loss)
            selected_total_size += self.clients[uid].n_item
        return collector, client_loss, selected_total_size

    def _check(self, iterator):
        assert abs(sum(iterator) - 1) <= 1e-4

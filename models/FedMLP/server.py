# from collections import OrderedDict
# from typing import Dict, List

# from models.base import ServerBase


# class Server(ServerBase):
#     """服务端只做模型参数的融合
#     """
#     def __init__(self) -> None:
#         super().__init__()
#         self._params = None

#     @property
#     def params(self):
#         return self._params

#     @params.setter
#     def params(self, params):
#         self._params = params

#     def __repr__(self) -> str:
#         return "Server()"

from collections import OrderedDict
from typing import Dict, List


class Server:
    """服务端只做模型参数的融合
    """
    def __init__(self) -> None:
        self.params = None

    def upgrade_wich_cefficients(self, params: List[Dict], coefficients: Dict):

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


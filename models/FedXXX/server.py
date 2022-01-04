from collections import OrderedDict
from typing import Dict, List


class Server:
    """服务端只做模型参数的融合
    """

    def __init__(self) -> None:
        self.params = None
        self.server_feature_map = {}

    def upgrade(self, params: List[Dict]):
        o = OrderedDict()
        if len(params) != 0:
            for k, v in params[0].items():
                o[k] = sum([i[k] for i in params]) / len(params)
            self.params = o

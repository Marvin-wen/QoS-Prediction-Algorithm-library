from collections import defaultdict
from numpy.lib.function_base import select
import numpy as np

class Client(object):
    def __init__(self, traid, uid, user_vec) -> None:
        super().__init__()
        self.traid = traid
        self.uid = uid
        self.n_item = len(traid)
        self.user_vec = user_vec


class Clients(object):
    def __init__(self, traid, n_user, latent_dim) -> None:
        super().__init__()
        self.traid = traid
        self.clients_map = {}
        self.users_vec = 2 * np.random.random((n_user,latent_dim)) - 1
        self.get_clients()

    def get_clients(self):
        s = set()
        r = defaultdict(list)
        for row in self.traid:
            uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
            r[uid].append(row)
        for uid,rows in r.items():
            self.clients_map[uid] = Client(np.array(rows),uid,self.users_vec[uid])
        print(f"Clients Nums:{len(self.clients_map)}")


import numpy as np


class Server(object):
    def __init__(self,n_item,latent_dim) -> None:
        super().__init__()
        self.n_item = n_item
        self.latent_dim = latent_dim
        self.items_vec = 2 * np.random.random((self.n_item,self.latent_dim)) - 1
    
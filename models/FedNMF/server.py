import numpy as np
from tqdm import tqdm


class Server(object):

    def __init__(self, n_item, latent_dim) -> None:
        super().__init__()
        self.n_item = n_item
        self.latent_dim = latent_dim
        # self.items_vec = 2 * np.random.random(
        # (self.n_item, self.latent_dim)) - 1
        self.items_vec = np.random.random((self.n_item, self.latent_dim))

    def upgrade(self, lr, gradient_from_user: list):
        """Server upgrades by user gradient
        """
        for gradient in gradient_from_user:
            iid, grad = gradient[0], gradient[1]
            self.items_vec[iid] -= lr * grad
            self.items_vec[self.items_vec < 0] = 0

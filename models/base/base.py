import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ModelBase(object):
    def __init__(self, model, loss_fn, use_gpu=True) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.device = ("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.tb = SummaryWriter()

    def fit(self, train_loader, epochs, optimizer, eval_=True, eval_loader=None):
        self.model.train()
        train_loss_list = []
        eval_loss_list = []
        self.optimizer = optimizer
        for epoch in tqdm(range(epochs)):
            train_batch_loss = 0
            eval_total_loss = 0
            for batch_id, batch in tqdm(enumerate(train_loader)):
                user, item, rating = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_real = rating.reshape(-1, 1)
                self.optimizer.zero_grad()
                y_pred = self.model(user, item)
                # must be (1. nn output, 2. target)
                loss = self.loss_fn(y_pred, y_real)
                loss.backward()
                self.optimizer.step()

                train_batch_loss += loss.item()

            loss_per_epoch = train_batch_loss / len(train_loader)
            train_loss_list.append(loss_per_epoch)
            print(
                f"Training Epoch:[{epoch}/{epochs}] Loss:{loss_per_epoch:.4f}")
            self.tb.add_scalar("Training Loss", loss_per_epoch, epoch)

            if eval_ and (epoch + 1) % 10 == 0:
                assert eval_loader is not None, "Please offer eval dataloader"
                self.model.eval()
                with torch.no_grad():
                    for batch_id, batch in tqdm(enumerate(eval_loader)):
                        user, item, rating = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                        y_pred = self.model(user, item)
                        y_real = rating.reshape(-1, 1)
                        loss = self.loss_fn(y_pred, y_real)
                        eval_total_loss += loss.item()
                    loss_per_epoch = eval_total_loss/len(eval_loader)
                    eval_loss_list.append(loss_per_epoch)
                    print(f"Test loss:", loss_per_epoch)
                    self.tb.add_scalar("Eval loss", loss_per_epoch, epoch)

    def predict(self):
        ...


class MemoryBase(object):
    def __init__(self) -> None:
        super().__init__()

    def fit():
        raise NotImplementedError

    def train():
        raise NotImplementedError

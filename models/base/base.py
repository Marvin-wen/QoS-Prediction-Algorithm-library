import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.model_util import save_checkpoint,load_checkpoint
from utils.evaluation import mae,mse,rmse

class ModelBase(object):
    def __init__(self, loss_fn, use_gpu=True) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.tb = SummaryWriter()
        self.device = ("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = self.__class__.__name__

    def fit(self, train_loader, epochs, optimizer, eval_=True, eval_loader=None, save_model=False, save_filename=""):
        """Eval 为True自动保存最优模型（推荐），save_model为True间隔epoch后自动保存模型

        Args:
            train_loader ([type]): [description]
            epochs ([type]): [description]
            optimizer ([type]): [description]
            eval_ (bool, optional): [description]. Defaults to True.
            eval_loader ([type], optional): [description]. Defaults to None.
            save_model (bool, optional): [description]. Defaults to True.
            model_name (str, optional): [description]. Defaults to "model_checkpoint.ckpt".
            max_keep (int, optional): [description]. Defaults to 10.
        """
        self.model.train()
        train_loss_list = []
        eval_loss_list = []
        best_loss = None
        is_best = False
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

            if  (epoch + 1) % 10 == 0:

                if eval_ == True:
                    assert eval_loader is not None, "Please offer eval dataloader"
                    self.model.eval()
                    with torch.no_grad():
                        for batch_id, batch in tqdm(enumerate(eval_loader)):
                            user, item, rating = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                            y_pred = self.model(user, item)
                            y_real = rating.reshape(-1, 1)
                            # loss = self.loss_fn(y_pred, y_real)
                            loss = nn.L1Loss()(y_pred,y_real)
                            eval_total_loss += loss.item()
                        loss_per_epoch = eval_total_loss/len(eval_loader)
                        if best_loss is None:
                            best_loss = loss_per_epoch
                            is_best = True
                        elif loss_per_epoch < best_loss:
                            best_loss = loss_per_epoch
                            is_best = True
                        else:
                            is_best = False
                        eval_loss_list.append(loss_per_epoch)
                        print(f"Test loss:", loss_per_epoch)
                        self.tb.add_scalar("Eval loss", loss_per_epoch, epoch)
                        if is_best:
                            ckpt = {
                                "model":self.model.state_dict(),
                                "epoch":epoch+1,
                                "optim":optimizer.state_dict(),
                                "best_loss":best_loss
                            }
                        else:
                            ckpt = {}
                        save_checkpoint(ckpt,is_best,f"output/{self.name}",f"loss_{best_loss:.4f}.ckpt")
                
                elif save_model:
                        ckpt = {
                            "model":self.model.state_dict(),
                            "epoch":epoch+1,
                            "optim":optimizer.state_dict(),
                            "best_loss":loss_per_epoch
                        }
                        save_checkpoint(ckpt,save_model,f"output/{self.name}",f"{save_filename}_loss_{loss_per_epoch:.4f}.ckpt")


    def predict(self,test_loader,resume=False,path=None):
        y_pred_list = []
        y_list = []
        if resume:
            ckpt = load_checkpoint(path)
            self.model.load_state_dict(ckpt['model'])
            print(f"last checkpoint restored! ckpt: loss {ckpt['best_loss']:.4f} Epoch {ckpt['epoch']}")
        
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in tqdm(enumerate(test_loader)):
                user, item, rating = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                y_pred = self.model(user, item)
                y_real = rating.reshape(-1, 1)
                y_pred_list.append(y_pred.squeeze())
                y_list.append(y_real.squeeze())

        return torch.cat(y_list),torch.cat(y_pred_list)



class MemoryBase(object):
    def __init__(self) -> None:
        super().__init__()

    def fit():
        raise NotImplementedError

    def train():
        raise NotImplementedError

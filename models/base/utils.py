import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_single_epoch_with_dataloader(model, device, dataloader: DataLoader,
                                       opt: Optimizer, loss_fn):
    """训练一个epoch

    Args:
        dataloader (DataLoader): [description]
        lr ([type]): [description]
        opt (Optimizer): [description]
        loss_fn (_Loss): [description]

    Returns: 返回一个epoch产生的loss,np.float64类型
    """
    loss_per_epoch = []
    for batch_id, batch in enumerate(dataloader):
        user, item, rating = batch[0].to(device), batch[1].to(
            device), batch[2].to(device)

        y_real = rating.reshape(-1, 1)
        opt.zero_grad()
        y_pred = model(user, item)
        loss = loss_fn(y_pred, y_real)
        loss.backward()
        opt.step()
        loss_per_epoch.append(loss.item())

    return np.average(loss_per_epoch)


def train_mult_epochs_with_dataloader(epochs, *args, **kwargs):
    train_loss_list = []
    for epoch in range(epochs):
        loss_per_eopch = train_single_epoch_with_dataloader(*args, **kwargs)
        train_loss_list.append(loss_per_eopch)
    return np.average(train_loss_list), train_loss_list

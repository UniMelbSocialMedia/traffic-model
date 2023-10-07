import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_array, max_epoch):
        self.lr_array = lr_array
        self.max_epoch = max_epoch
        super().__init__(optimizer)

    def get_lr(self):
        index = self.last_epoch % self.max_epoch
        return [self.lr_array[index]] * len(self.optimizer.param_groups)
        # if self.last_epoch < len(self.lr_array):
        #     return [self.lr_array[self.last_epoch]] * len(self.optimizer.param_groups)
        # else:
        #     return [self.lr_array[-1]] * len(self.optimizer.param_groups)

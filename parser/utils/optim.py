# -*- coding: utf-8 -*-

from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        scale = min(epoch ** (-0.5),
                    epoch * self.warmup_steps ** (-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]

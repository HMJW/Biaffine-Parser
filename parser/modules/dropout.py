# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask


class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items, eps=1e-12):
        if self.training:
            masks = [torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
                     for x in items]
            scale = len(items) / (sum(masks) + eps)
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items

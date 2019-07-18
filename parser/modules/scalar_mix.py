# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ScalarMix(nn.Module):

    def __init__(self, n_layers):
        super(ScalarMix, self).__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))

    def extra_repr(self):
        return f"n_layers={self.n_layers}"

    def forward(self, tensors):
        normed_weights = self.weights.softmax(dim=0)
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum

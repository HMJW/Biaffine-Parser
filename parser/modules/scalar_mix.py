# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ScalarMix(nn.Module):

    def __init__(self, n_layers, do_layer_norm=False):
        super(ScalarMix, self).__init__()

        self.n_layers = n_layers
        self.do_layer_norm = do_layer_norm

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        if self.do_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(800)
                                              for _ in range(n_layers)])

    def extra_repr(self):
        s = f"n_layers={self.n_layers}"
        if self.do_layer_norm:
            s += f", do_layer_norm={self.do_layer_norm}"

        return s

    def forward(self, tensors, mask=None):
        normed_weights = self.weights.softmax(dim=0)

        if self.do_layer_norm:
            tensors = [self.layer_norms[i](h) for i, h in enumerate(tensors)]
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum

# -*- coding: utf-8 -*-

from parser.modules.dropout import SharedDropout
from parser.modules.transformer import Layer

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_packed_sequence)


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        self.attentive_layers = nn.ModuleList()
        for layer in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            if layer < self.num_layers - 1:
                self.attentive_layers.append(Layer(8,
                                                   hidden_size * 2,
                                                   50,
                                                   1200))
            input_size = hidden_size * 2
        self.dropout = SharedDropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.f_cells:
            for i in cell.parameters():
                if len(i.shape) > 1:
                    nn.init.orthogonal_(i)
                else:
                    nn.init.zeros_(i)
        for cell in self.b_cells:
            for i in cell.parameters():
                if len(i.shape) > 1:
                    nn.init.orthogonal_(i)
                else:
                    nn.init.zeros_(i)

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        if self.training:
            hid_mask = SharedDropout.get_mask(h, self.p)

        for t in steps:
            last_batch_size, batch_size = len(h), batch_sizes[t]
            if last_batch_size < batch_size:
                h = torch.cat((h, init_h[last_batch_size:batch_size]))
                c = torch.cat((c, init_c[last_batch_size:batch_size]))
            else:
                h = h[:batch_size]
                c = c[:batch_size]
            h, c = cell(x[t], (h, c))
            output.append(h)
            if self.training:
                h = h * hid_mask[:batch_size]
        if reverse:
            output.reverse()
        output = torch.cat(output)

        return output

    def forward(self, x, mask, hx=None):
        residual = x
        lens = mask.sum(1)
        sequence = pack_padded_sequence(x, lens, True)
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        max_batch_size = batch_sizes[0]

        if hx is None:
            init = x.new_zeros(max_batch_size, self.hidden_size)
            hx = (init, init)

        for layer in range(self.num_layers):
            x = pack_padded_sequence(residual, lens, True).data
            if self.training:
                seq_mask = SharedDropout.get_mask(x[:max_batch_size], self.p)
                seq_mask = torch.cat([seq_mask[:batch_size]
                                      for batch_size in batch_sizes])
                x *= seq_mask
            x = torch.split(x, batch_sizes)
            f_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.f_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=False)
            b_output = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.b_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=True)
            x = torch.cat([f_output, b_output], -1)
            x = pad_packed_sequence(PackedSequence(x, sequence.batch_sizes),
                                    True)[0]
            if layer < self.num_layers - 1:
                x = self.dropout(x)
                if layer == 0:
                    x = self.layer_norm(x)
                else:
                    x = self.layer_norm(x + residual)
                x = self.attentive_layers[layer](x, mask)
            residual = x

        return x

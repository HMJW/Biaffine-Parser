# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, n_layers, n_heads, n_model, n_hidden, p=0.2):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([
            Layer(n_heads, n_model, n_hidden, p)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(p)

    def init_pos(self, x):
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)) // 2 * 2 / n_model)
        pos[:, 0::2] = pos[:, 0::2].sin()
        pos[:, 1::2] = pos[:, 1::2].cos()

        return pos

    def forward(self, x, mask):
        x += self.init_pos(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
            yield x

        return x


class Layer(nn.Module):

    def __init__(self, n_heads, n_model, n_hidden, p=0.2):
        super(Layer, self).__init__()

        self.attn = MultiheadAttention(n_heads, n_model//n_heads, n_model, p)
        self.ffn = PosWiseFFN(n_model, n_hidden, p)

    def forward(self, x, mask):
        x = self.attn(x, x, x, mask)
        x = self.ffn(x)

        return x


class MultiheadAttention(nn.Module):

    def __init__(self, n_heads, n_embed, n_model, p=0.2):
        super(MultiheadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_model = n_model
        self.n_embed = n_embed
        self.scale = n_embed ** 0.5

        self.wq = nn.Parameter(torch.Tensor(n_heads, n_model, n_embed))
        self.wk = nn.Parameter(torch.Tensor(n_heads, n_model, n_embed))
        self.wv = nn.Parameter(torch.Tensor(n_heads, n_model, n_embed))

        self.proj = nn.Linear(n_heads * n_embed, n_model, False)
        self.norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(p)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.wq)
        nn.init.xavier_normal_(self.wk)
        nn.init.xavier_normal_(self.wv)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, Q, K, V, mask):
        residual = Q
        batch_size, seq_len, n_model = Q.shape

        # [n_heads * batch_size, seq_len, n_embed]
        Q = (Q @ self.wq.unsqueeze(1)).view(-1, seq_len, self.n_embed)
        # [n_heads * batch_size, seq_len, n_embed]
        K = (K @ self.wk.unsqueeze(1)).view(-1, seq_len, self.n_embed)
        # [n_heads * batch_size, seq_len, n_embed]
        V = (V @ self.wv.unsqueeze(1)).view(-1, seq_len, self.n_embed)

        # Scaled Dot-Product Attention
        # [n_heads * batch_size, 1, seq_len]
        mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
        # [n_heads * batch_size, seq_len, seq_len]
        attn = (Q @ K.transpose(-1, -2)) / self.scale
        attn = attn.masked_fill(~mask, float('-inf')).softmax(-1)
        attn = self.dropout(attn)

        # [n_heads * batch_size, seq_len, n_embed]
        x = attn @ V
        # [batch_size, seq_len, n_heads * n_embed]
        x = torch.cat(torch.split(x, batch_size, 0), -1)
        # [batch_size, seq_len, n_model]
        x = self.proj(x)
        x = self.dropout(x)
        x = self.norm(x + residual)

        return x


class PosWiseFFN(nn.Module):

    def __init__(self, n_model, n_hidden, p=0.2):
        super(PosWiseFFN, self).__init__()

        self.w1 = nn.Sequential(nn.Linear(n_model, n_hidden), nn.ReLU())
        self.w2 = nn.Linear(n_hidden, n_model)
        self.norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        residual = x
        x = self.w1(x)
        x = self.w2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)

        return x

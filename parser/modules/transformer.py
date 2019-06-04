# -*- coding: utf-8 -*-

import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, n_layers, n_heads, n_model, n_embed, n_inner,
                 input_dropout=0.1, attn_dropout=0.1, ffn_dropout=0.1):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([Layer(n_heads, n_model, n_embed, n_inner,
                                           attn_dropout, ffn_dropout)
                                     for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(input_dropout)

    def forward(self, x, mask):
        x += self.init_pos(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

    @classmethod
    def init_pos(cls, x):
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len)).unsqueeze(-1)
        pos = pos / 10000 ** (x.new_tensor(range(n_model)) // 2 * 2 / n_model)
        pos[:, 0::2] = pos[:, 0::2].sin()
        pos[:, 1::2] = pos[:, 1::2].cos()
        pos = pos.unsqueeze(0).expand_as(x)

        return pos


class Layer(nn.Module):

    def __init__(self, n_heads, n_model, n_embed, n_inner,
                 attn_dropout=0.1, ffn_dropout=0.1):
        super(Layer, self).__init__()

        self.attn = MultiHeadAttention(n_heads, n_model, n_embed, attn_dropout)
        self.ffn = PosWiseFFN(n_model, n_inner, ffn_dropout)

    def forward(self, x, mask):
        x = self.attn(x, x, x, mask)
        x = self.ffn(x)

        return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, scale, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask):
        attn = (q @ k.transpose(-1, -2)) / self.scale
        attn = attn.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        x = attn @ v

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, n_model, n_embed, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_model = n_model
        self.n_embed = n_embed

        self.wq = nn.Linear(n_model, n_heads*n_embed, False)
        self.wk = nn.Linear(n_model, n_heads*n_embed, False)
        self.wv = nn.Linear(n_model, n_heads*n_embed, False)
        self.attn = ScaledDotProductAttention(n_embed**0.5, dropout)
        self.layer_norm = nn.LayerNorm(n_model)
        self.wo = nn.Linear(n_heads*n_embed, n_model, False)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.wq.weight)
        nn.init.orthogonal_(self.wk.weight)
        nn.init.orthogonal_(self.wv.weight)

    def forward(self, q, k, v, mask):
        residual = q
        batch_size, seq_len, _ = q.shape

        # [batch_size, seq_len, n_heads, n_embed]
        q = self.wq(q).view(batch_size, seq_len, self.n_heads, -1)
        # [batch_size, seq_len, n_heads, n_embed]
        k = self.wk(k).view(batch_size, seq_len, self.n_heads, -1)
        # [batch_size, seq_len, n_heads, n_embed]
        v = self.wv(v).view(batch_size, seq_len, self.n_heads, -1)
        # [n_heads * batch_size, seq_len, n_embed]
        q = q.permute(2, 0, 1, 3).reshape(-1, seq_len, self.n_embed)
        # [n_heads * batch_size, seq_len, n_embed]
        k = k.permute(2, 0, 1, 3).reshape(-1, seq_len, self.n_embed)
        # [n_heads * batch_size, seq_len, n_embed]
        v = v.permute(2, 0, 1, 3).reshape(-1, seq_len, self.n_embed)

        # [n_heads * batch_size, seq_len, n_embed]
        x = self.attn(q, k, v, mask.repeat(self.n_heads, 1))
        x = x.view(self.n_heads, batch_size, seq_len, self.n_embed)
        # [batch_size, seq_len, n_heads * n_embed]
        x = x.permute(1, 2, 0, 3).reshape(batch_size, seq_len, -1)
        # [batch_size, seq_len, n_model]
        x = self.wo(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


class PosWiseFFN(nn.Module):

    def __init__(self, n_model, n_inner, p=0.1):
        super(PosWiseFFN, self).__init__()

        self.w1 = nn.Linear(n_model, n_inner)
        self.activation = nn.ReLU()
        self.w2 = nn.Linear(n_inner, n_model)
        self.layer_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(p)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.w1.weight)
        nn.init.orthogonal_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x):
        residual = x
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x

# -*- coding: utf-8 -*-

from parser.modules import (CHAR_LSTM, MLP, Biaffine, BiLSTM,
                            IndependentDropout, ScalarMix, SharedDropout,
                            Transformer)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class BiaffineParser(nn.Module):

    def __init__(self, config, embeddings):
        super(BiaffineParser, self).__init__()

        self.config = config
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embeddings)
        self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                       embedding_dim=config.n_embed)
        # the char-lstm layer
        self.char_lstm = CHAR_LSTM(n_chars=config.n_chars,
                                   n_embed=config.n_char_embed,
                                   n_out=config.n_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        # the word-lstm layer
        self.transformer = Transformer(n_layers=6,
                                       n_heads=8,
                                       n_model=200,
                                       n_hidden=400)
        self.scalar_mix = ScalarMix(6)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=200,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=200,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=200,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=200,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=config.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embed.weight)

    def forward(self, words, chars):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.pretrained(words) + self.word_embed(ext_words)
        char_embed = self.char_lstm(chars[mask])
        char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)
        word_embed, char_embed = self.embed_dropout(word_embed, char_embed)
        # concatenate the word and char representations
        x = torch.cat((word_embed, char_embed), dim=-1)
        x = self.scalar_mix([i for i in self.transformer(x, mask)])
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embeddings'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'embeddings': self.pretrained.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)

# -*- coding: utf-8 -*-

from parser.modules import (CHAR_LSTM, MLP, Biaffine, BiLSTM,
                            IndependentDropout, SharedDropout, BertEmbedding)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class BiaffineScorer(nn.Module):

    def __init__(self, input_dim, n_mlp_arc, n_mlp_rel, n_rels, mlp_dropout=0):
        super(BiaffineScorer, self).__init__()
        self.mlp_arc_h = MLP(n_in=input_dim,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=input_dim,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=input_dim,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=input_dim,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
    
    def forward(self, x, mask):
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


class BiaffineParser(nn.Module):

    def __init__(self, config, embed):
        super(BiaffineParser, self).__init__()

        self.config = config
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embed)
        self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                       embedding_dim=config.n_embed)
        # the char-lstm layer
        self.char_lstm = CHAR_LSTM(n_chars=config.n_chars,
                                   n_embed=config.n_char_embed,
                                   n_out=config.n_embed)
        self.bert = BertEmbedding(config.bert_model, config.bert_layer)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=config.n_embed*2 + self.bert.hidden_size,
                           hidden_size=config.n_lstm_hidden,
                           num_layers=config.n_lstm_layers,
                           dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)
        self.task = config.task.split()
        self.n_task = len(self.task)

        self.scorers = nn.ModuleList([BiaffineScorer(config.n_lstm_hidden*2, 
                                                    config.n_mlp_arc, 
                                                    config.n_mlp_rel, 
                                                    rel_size, 
                                                    config.mlp_dropout) 
                                                    for rel_size in config.n_rels])


        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embed.weight)

    def forward(self, words, chars, subwords, sub_masks, sub_lens, tasks):
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
        bert_embed = self.bert(subwords, sub_lens, sub_masks)
        word_embed, char_embed, bert_embed = self.embed_dropout(word_embed, char_embed, bert_embed)
        # concatenate the word and char representations
        x = torch.cat((word_embed, char_embed, bert_embed), dim=-1)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]

        s_arcs, s_rels = [], []
        for i in range(self.n_task):
            task_mask = tasks.eq(i)
            if task_mask.sum() == 0:
                s_arcs.append(None)
                s_rels.append(None)
            else:
                task_x = x[task_mask]
                task_mask = mask[task_mask]
                s_arc, s_rel = self.scorers[i](task_x, task_mask)
                s_arcs.append(s_arc)
                s_rels.append(s_rel)

        return s_arcs, s_rels

    @classmethod
    def load(cls, fname):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embed'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'embed': self.pretrained.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)

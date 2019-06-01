# -*- coding: utf-8 -*-

from parser.modules import MLP, Biaffine, IndependentDropout, Transformer

import torch
import torch.nn as nn


class BiaffineParser(nn.Module):

    def __init__(self, config, embeddings):
        super(BiaffineParser, self).__init__()

        self.config = config
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embeddings)
        self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                       embedding_dim=config.n_embed)
        self.tag_embed = nn.Embedding(num_embeddings=config.n_tags,
                                      embedding_dim=config.n_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)
        self.projection = nn.Linear(in_features=config.n_embed*2,
                                    out_features=config.n_model,
                                    bias=False)

        self.transformer = Transformer(n_layers=config.n_layers,
                                       n_heads=config.n_heads,
                                       n_model=config.n_model,
                                       n_embed=config.n_model//config.n_heads,
                                       n_inner=config.n_inner,
                                       p=config.encoder_dropout)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=config.n_model,
                             n_out=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=config.n_model,
                             n_out=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.n_model,
                             n_out=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.n_model,
                             n_out=config.n_mlp_rel,
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

    def forward(self, words, tags):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.pretrained(words) + self.word_embed(ext_words)
        tag_embed = self.tag_embed(tags)
        word_embed, tag_embed = self.embed_dropout(word_embed, tag_embed)
        # concatenate the word and tag representations
        embed = torch.cat((word_embed, tag_embed), dim=-1)

        x = self.projection(embed)
        x = self.transformer(x, mask)

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

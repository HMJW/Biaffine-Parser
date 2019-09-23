# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .modules import (MLP, Biaffine, BiLSTM, IndependentDropout,
                         SharedDropout)
from shared import Base, save_config, get_config
from .utils import arc_argmax, rel_argmax, Viterbi

class Dep(Base):

    def __init__(self, vocab, config, embeddings):
        super(Dep, self).__init__()
        self.config = config
        self.vocab = vocab
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embeddings)
        self.embed = nn.Embedding(num_embeddings=vocab.n_train_words,
                                  embedding_dim=config.n_embed)
        self.tag_embed = nn.Embedding(num_embeddings=vocab.n_tags,
                                      embedding_dim=config.n_tag_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=config.n_embed+config.n_tag_embed,
                           hidden_size=config.n_lstm_hidden,
                           num_layers=config.n_lstm_layers,
                           dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # the MLP layers
        self.mlp_arc_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=vocab.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = vocab.pad_index
        self.unk_index = vocab.unk_index

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.embed.weight)

    def forward(self, words, tags):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        embed = self.pretrained(words) + self.embed(ext_words)
        tag_embed = self.tag_embed(tags)
        embed, tag_embed = self.embed_dropout(embed, tag_embed)
        # concatenate the word and tag representations
        x = torch.cat((embed, tag_embed), dim=-1)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]

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
    def load(cls, path):
        vocab_path = os.path.join(path, "vocab.pt")
        param_path = os.path.join(path, "param.pt")
        ext_emb_path = os.path.join(path, "ext_word_emb.pt")
        config_path = os.path.join(path, "config.json")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        config = get_config(config_path)

        vocab = torch.load(vocab_path)
        ext_word_emb = torch.load(ext_emb_path, map_location=device)
        network = cls(vocab, config, ext_word_emb)

        state = torch.load(param_path, map_location=device)
        network.load_state_dict(state)
        network.to(device)
        return network

    def save(self, path):
        vocab_path = os.path.join(path, "vocab.pt")
        param_path = os.path.join(path, "param.pt")
        ext_emb_path = os.path.join(path, "ext_word_emb.pt")
        config_path = os.path.join(path, "config.json")

        torch.save(self.vocab, vocab_path)
        save_config(self.config, config_path)
        torch.save(self.pretrained.weight, ext_emb_path)
        torch.save(self.state_dict(), param_path)

    @torch.no_grad()
    def predict(self, word_list, pos_list, partial_labels=None):
        assert len(word_list) == len(pos_list)
        self.eval()

        word_list = [self.vocab.ROOT] + word_list
        pos_list = [self.vocab.ROOT] + pos_list

        word_idxs = self.vocab.word2id(word_list)
        pos_idxs = self.vocab.tag2id(pos_list)

        word_idxs = word_idxs.unsqueeze(0)
        pos_idxs = pos_idxs.unsqueeze(0)
        if torch.cuda.is_available():
            word_idxs = word_idxs.cuda()
            pos_idxs = pos_idxs.cuda()

        length = len(word_list)
        s_arc, s_rel = self.forward(word_idxs, pos_idxs)
        s_arc, s_rel = s_arc.squeeze(0), s_rel.squeeze(0)
        arc_probs = F.softmax(s_arc, dim=-1)
        rel_probs = F.softmax(s_rel, dim=-1)
        
        pred_arcs = arc_argmax(arc_probs.data.numpy(), length, ensure_tree=True)
        pred_probs = arc_probs[torch.arange(length), pred_arcs]
        
        if "ROOT" in self.vocab.rel_dict:
            root_id = self.vocab.rel_dict["ROOT"]
        else:
            root_id = self.vocab.rel_dict["root"]
        # pred_arcs = s_arc.argmax(dim=-1)

        rel_probs = rel_probs[torch.arange(length), pred_arcs]
        pred_rels = rel_argmax(rel_probs, length, root_id, ensure_tree=True)
        pred_rels = self.vocab.id2rel(pred_rels)
        return pred_arcs[1:].tolist(), pred_probs.tolist(), pred_rels[1:]

    @torch.no_grad()
    def predict_batch(self, word_list, pos_list, partial_labels=None):
        assert len(word_list) == len(pos_list)
        self.eval()

        word_idxs = [self.vocab.word2id([self.vocab.ROOT] + word) for word in word_list]
        pos_idxs = [self.vocab.tag2id([self.vocab.ROOT] + p) for p in pos_list]

        if torch.cuda.is_available():
            word_idxs = pad_sequence(word_idxs, True).cuda()
            pos_idxs = pad_sequence(pos_idxs, True).cuda()
        else:
            word_idxs = pad_sequence(word_idxs, True)
            pos_idxs = pad_sequence(pos_idxs, True)

        s_arc, s_rel = self.forward(word_idxs, pos_idxs)
        mask = word_idxs.ne(0)
        lens = mask.sum(1).tolist()
        arc_probs = F.softmax(s_arc, dim=-1)
        rel_probs = F.softmax(s_rel, dim=-1)
        
        pred_arcs, pred_rels, pred_probs = [], [], []
        if "ROOT" in self.vocab.rel_dict:
            root_id = self.vocab.rel_dict["ROOT"]
        else:
            root_id = self.vocab.rel_dict["root"]
        for arc, rel, length in zip(arc_probs, rel_probs, lens):
            arc = arc[:length, :length]
            rel = rel[:length, :length]

            pred_arc, pred_rel = Viterbi.decode_one_inst(arc, rel, length)
            # pred_arc = arc_argmax(arc.data.numpy(), length, ensure_tree=True)
            pred_prob = arc[torch.arange(length), pred_arc]
            
            # pred_arcs = s_arc.argmax(dim=-1)
            # rel_prob = rel[torch.arange(length), pred_arc]
            # pred_rel = rel_argmax(rel_prob, length, root_id, ensure_tree=True)
            pred_rel = self.vocab.id2rel(pred_rel)

            pred_arcs.append(pred_arc[1:].tolist())
            pred_probs.append(pred_prob[1:].tolist())
            pred_rels.append(pred_rel[1:])
        return pred_arcs, pred_probs, pred_rels
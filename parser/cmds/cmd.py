# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import eisner
from parser.utils.common import bos, pad, unk
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import BertField, CharField, Field
from parser.utils.fn import ispunct, numericalize, pdeprel_numericalize
from parser.utils.metric import Metric

import torch
import torch.nn as nn
from transformers import BertTokenizer


class CMD(object):

    def __call__(self, args):
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            self.WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            self.PDEPREL = Field('pdeprels', use_vocab=False, fn=pdeprel_numericalize)
            if args.feat == 'char':
                self.FEAT = CharField('chars', pad=pad, unk=unk, bos=bos,
                                      fix_len=args.fix_len, tokenize=list)
            elif args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert', pad='[PAD]', bos='[CLS]',
                                      tokenize=tokenizer.encode)
            else:
                self.FEAT = Field('tags', bos=bos)
            self.HEAD = Field('heads', bos=bos, use_vocab=False,
                              fn=numericalize)
            self.REL = Field('rels', bos=bos)
            if args.feat in ('char', 'bert'):
                self.fields = CoNLL(FORM=(self.WORD, self.FEAT),
                                    HEAD=self.HEAD, DEPREL=self.REL, PDEPREL=self.PDEPREL)
            else:
                self.fields = CoNLL(FORM=self.WORD, CPOS=self.FEAT,
                                    HEAD=self.HEAD, DEPREL=self.REL, PDEPREL=self.PDEPREL)

            train = Corpus.load(args.ftrain, self.fields)
            if args.fembed:
                embed = Embedding.load(args.fembed, args.unk)
            else:
                embed = None
            self.WORD.build(train, args.min_freq, embed)
            self.FEAT.build(train)
            self.REL.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat in ('char', 'bert'):
                self.WORD, self.FEAT = self.fields.FORM
            else:
                self.WORD, self.FEAT = self.fields.FORM, self.fields.CPOS
            self.HEAD, self.REL = self.fields.HEAD, self.fields.DEPREL
        self.puncts = torch.tensor([i for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(args.device)
        self.criterion = nn.CrossEntropyLoss()

        print(f"{self.WORD}\n{self.FEAT}\n{self.HEAD}\n{self.REL}")
        args.update({
            'n_words': self.WORD.vocab.n_init,
            'n_feats': len(self.FEAT.vocab),
            'n_rels': len(self.REL.vocab),
            'pad_index': self.WORD.pad_index,
            'unk_index': self.WORD.unk_index,
            'bos_index': self.WORD.bos_index
        })

    def train(self, loader):
        self.model.train()

        for words, feats, arcs, rels, _ in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            arc_scores, rel_scores = self.model(words, feats)
            loss = self.get_loss(arc_scores, rel_scores, arcs, rels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        loss, metric = 0, Metric()

        for words, feats, arcs, rels, _ in loader:
            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            arc_scores, rel_scores = self.model(words, feats)
            loss += self.get_loss(arc_scores, rel_scores, arcs, rels, mask)
            arc_preds, rel_preds, _ = self.decode(arc_scores, rel_scores, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_arcs, all_rels, all_probs = [], [], []
        for words, feats in loader:
            mask = words.ne(self.args.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            arc_scores, rel_scores = self.model(words, feats)
            arc_preds, rel_preds, prob_preds = self.decode(arc_scores, rel_scores, mask)
            all_arcs.extend(arc_preds[mask].split(lens))
            all_rels.extend(rel_preds[mask].split(lens))
            all_probs.extend(prob_preds[mask].split(lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.REL.vocab.id2token(seq.tolist()) for seq in all_rels]
        all_probs = [seq.tolist() for seq in all_probs]
        return all_arcs, all_rels, all_probs

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask):
        if self.args.partial:
            mask &= arcs.ge(0)
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(arc_scores, arcs)
        rel_loss = self.criterion(rel_scores, rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, arc_scores, rel_scores, mask):
        if self.args.tree:
            arc_preds = eisner(arc_scores, mask)
        else:
            arc_preds = arc_scores.argmax(-1)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        p_arc, p_rel = arc_scores.softmax(dim=-1), rel_scores.softmax(dim=-1)
        p_arc = p_arc.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        p_rel = p_rel.max(-1)[0]
        p_rel = p_rel.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        p = p_arc * p_rel
        return arc_preds, rel_preds, p

# -*- coding: utf-8 -*-

from parser.metric import Metric
from parser.utils.alg import crf, eisner
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class Model(object):

    def __init__(self, config, vocab, parser):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab
        self.parser = parser
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def train(self, loader):
        self.parser.train()
        # max_batch = max(len(l) for l in loader)
        # loaders = [iter(l) for l in loader]
        
        # for _ in range(max_batch):
        #     self.optimizer.zero_grad()

        #     for i in range(len(loaders)):
        #         try:
        #             words, chars, subwords, bert_masks, lens, tasks, arcs, rels = next(loaders[i])
        #         except StopIteration as s:
        #             loaders[i] = iter(loader[i])
        #             words, chars, subwords, bert_masks, lens, tasks ,arcs, rels = next(loaders[i])
        #         mask = words.ne(self.vocab.pad_index)
        #         mask[:, 0] = 0
        #         s_arcs, s_rels = self.parser(words, chars, subwords, bert_masks, lens, tasks)
        #         for i, (s_arc, s_rel) in enumerate(zip(s_arcs, s_rels)):
        #             if s_arc is None or s_rel is None:
        #                 continue
        #             task_mask = tasks.eq(i)
        #             current_mask = mask[task_mask]
        #             gold_arcs, gold_rels = arcs[task_mask], rels[task_mask]
        #             arc_loss, _ = self.get_arc_loss(s_arc, gold_arcs, mask)
        #             rel_loss = self.get_rel_loss(s_rel, gold_arcs, gold_rels, current_mask)
        #             loss = (arc_loss + rel_loss) / current_mask.sum()
        #             loss.backward()
        #         nn.utils.clip_grad_norm_(self.parser.parameters(),
        #                     self.config.clip)
        #         self.optimizer.step()
        #         self.scheduler.step()

        for words, chars, subwords, bert_masks, lens, tasks, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arcs, s_rels = self.parser(words, chars, subwords, bert_masks, lens, tasks)

            batch_loss = 0
            batch_size, word_num = mask.sum().float(), mask.size(0)

            batch_gold_arcs, batch_s_arcs, batch_mask = [], [], []
            for i, (s_arc, s_rel) in enumerate(zip(s_arcs, s_rels)):
                if s_arc is None or s_rel is None:
                    continue
                task_mask = tasks.eq(i)
                current_mask = mask[task_mask]
                gold_arcs, gold_rels = arcs[task_mask], rels[task_mask]

                batch_gold_arcs.append(gold_arcs)
                batch_mask.append(current_mask)
                batch_s_arcs.append(s_arc)
                rel_loss = self.get_rel_loss(s_rel, gold_arcs, gold_rels, current_mask)
                batch_loss = batch_loss + rel_loss

            batch_gold_arcs = torch.cat(batch_gold_arcs, dim=0)
            batch_s_arcs = torch.cat(batch_s_arcs, dim=0)
            batch_mask = torch.cat(batch_mask, dim=0)

            batch_arc_loss, _ = self.get_arc_loss(batch_s_arcs, batch_gold_arcs, batch_mask)
            batch_loss = batch_loss / word_num + batch_arc_loss / word_num
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.parser.parameters(),
                                     self.config.clip)
            self.optimizer.step()
            self.scheduler.step()


    @torch.no_grad()
    def evaluate(self, loader, punct=False):
        self.parser.eval()

        total_loss, metric = 0, Metric()
        word_num = 0
        
        for words, chars, subwords, bert_masks, lens, tasks, arcs, rels in loader:
            puncts = words.new_tensor(self.vocab.puncts)
            puncts_mask = words.unsqueeze(-1).ne(puncts).all(-1)
            
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            word_num += mask.sum()
            s_arcs, s_rels = self.parser(words, chars, subwords, bert_masks, lens, tasks)
            for i, (s_arc, s_rel) in enumerate(zip(s_arcs, s_rels)):
                if s_arc is None and s_rel is None:
                    continue

                task_mask = tasks.eq(i)
                current_mask = mask[task_mask]
                gold_arcs, gold_rels = arcs[task_mask], rels[task_mask]

                if len(s_arc) > 0:
                    loss, s_arc = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels, current_mask)
                    total_loss += loss
                    if self.config.marg:
                        pred_arcs, pred_rels = self.decode(s_arc, s_rel, current_mask)
                    else:
                        pred_arcs, pred_rels = self.decode(s_arc.softmax(-1), s_rel, current_mask)

                    current_mask =  current_mask & gold_arcs.ge(0)
                    current_mask =  current_mask & puncts_mask[task_mask]

                    gold_arcs, gold_rels = gold_arcs[current_mask], gold_rels[current_mask]
                    pred_arcs, pred_rels = pred_arcs[current_mask], pred_rels[current_mask]
                    metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

        total_loss /= word_num
        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_arcs, all_rels = [], []
        for words, chars, subwords, bert_masks, lens, tasks in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_arcs, s_rels = self.parser(words, chars, subwords, bert_masks, lens, tasks)
            s_arc, s_rel = s_arcs[tasks[0]], s_rels[tasks[0]]

            if self.config.marg:
                s_arc = crf(s_arc, mask)
            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask)

            all_arcs.extend(torch.split(pred_arcs[mask], lens))
            all_rels.extend(torch.split(pred_rels[mask], lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq, tasks[0]) for seq in all_rels]

        return all_arcs, all_rels

    def get_arc_loss(self, s_arc, gold_arcs, mask):
        if not self.config.crf:
            mask = mask & gold_arcs.ge(0)
            s_arc = s_arc[mask]
            gold_arcs = gold_arcs[mask]

            arc_loss = self.criterion(s_arc, gold_arcs)
            return arc_loss, s_arc
        else:
            arc_loss, arc_probs = crf(s_arc, mask, gold_arcs,
                                    self.config.partial)
            return arc_loss, arc_probs

    def get_rel_loss(self, s_rel, gold_arcs, gold_rels, mask):
        mask = mask & gold_arcs.ge(0)
        s_rel = s_rel[mask]
        gold_rels = gold_rels[mask]
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs[mask]]

        rel_loss = self.criterion(s_rel, gold_rels)
        return rel_loss

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels, mask):
        if not self.config.crf:
            mask = mask & gold_arcs.ge(0)
            s_arcs, s_rels = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = gold_arcs[mask], gold_rels[mask]
            s_rels = s_rels[torch.arange(len(s_rels)), gold_arcs]

            arc_loss = self.criterion(s_arcs, gold_arcs)
            rel_loss = self.criterion(s_rels, gold_rels)
            return arc_loss + rel_loss, s_arc
        else:
            arc_loss, arc_probs = crf(s_arc, mask, gold_arcs,
                                    self.config.partial)
            mask = mask & gold_arcs.ge(0)
            s_rel, gold_rels = s_rel[mask], gold_rels[mask]
            s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs[mask]]
            rel_loss = self.criterion(s_rel, gold_rels)
            return arc_loss + rel_loss, arc_probs
        
    def decode(self, s_arc, s_rel, mask):
        if self.config.tree:
            arc_preds = eisner(s_arc, mask)
        else:
            arc_preds = s_arc.argmax(-1)
        rel_preds = s_rel.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds

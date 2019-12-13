# -*- coding: utf-8 -*-

import unicodedata
from collections import Counter

import torch


class Vocab(object):
    pad = '<pad>'
    unk = '<unk>'

    def __init__(self, words, chars, rels, task):
        self.pad_index = 0
        self.unk_index = 1
        self.task = task
        self.words = [self.pad, self.unk] + sorted(words)
        self.chars = [self.pad, self.unk] + sorted(chars)
        self.rels = [sorted(r) for r in rels] 

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.rel_dict = [{rel: i for i, rel in enumerate(r)} for r in self.rels]

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if self.is_punctuation(word))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_rels = [len(r) for r in self.rels]
        self.n_init = self.n_words

    def __repr__(self):
        s = f"{self.__class__.__name__}: "
        s += f"{self.task}: "
        s += f"{self.n_words} words, "
        s += f"{self.n_chars} chars, "
        s += f"{self.n_rels} rels"

        return s

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def char2id(self, sequence, max_length=20):
        char_ids = torch.zeros(len(sequence), max_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:max_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids

    def rel2id(self, sequence, task):
        return torch.tensor([self.rel_dict[task].get(rel, 0)
                             for rel in sequence])

    def id2rel(self, ids, task):
        return [self.rels[task][i] for i in ids]

    def read_embeddings(self, embed, smooth=True):
        words = [word.lower() for word in embed.tokens]
        # if the `unk` token has existed in the pretrained,
        # then replace it with a self-defined one
        if embed.unk:
            words[embed.unk_index] = self.unk

        self.extend(words)
        self.embed = torch.zeros(self.n_words, embed.dim)
        self.embed[self.word2id(words)] = embed.vectors

        if smooth:
            self.embed /= torch.std(self.embed)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.chars += sorted(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if self.is_punctuation(word))
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus, training=True):
        words = [self.word2id(seq) for seq in corpus.words]
        chars = [self.char2id(seq) for seq in corpus.words]
        tasks = [task for task in corpus.tasks]
        if not training:
            return words, chars, tasks
        arcs = [torch.tensor(seq) for seq in corpus.heads]

        rels = [self.rel2id(seq, task) for seq, task in zip(corpus.rels, corpus.tasks)]
        assert len(words) == len(chars) == len(arcs) == len(rels) == len(tasks)
        return words, chars, arcs, rels, tasks

    @classmethod
    def from_corpus(cls, corpus, min_freq=1, task=[]):
        words = Counter(word.lower() for c in corpus for seq in c.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for c in corpus for seq in c.words for char in ''.join(seq)})
        assert len(task) > 0
        rels = [list({rel for seq in c.rels for rel in seq}) for c in corpus]
        vocab = cls(words, chars, rels, task)

        return vocab

    @classmethod
    def is_punctuation(cls, word):
        return all(unicodedata.category(char).startswith('P') for char in word)

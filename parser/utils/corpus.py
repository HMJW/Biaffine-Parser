# -*- coding: utf-8 -*-

from collections import namedtuple
from parser.utils.fn import isprojective

Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL', 'TASK'],
                      defaults=[None]*11)


class Corpus(object):
    root = '<ROOT>'

    def __init__(self, sentences, task=0):
        super(Corpus, self).__init__()
        self.task = task
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i))
                      for i in zip(*(f for f in sentence[:-1] if f))) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        return [[self.root] + list(sentence.FORM) for sentence in self]

    @property
    def tags(self):
        return [[self.root] + list(sentence.CPOS) for sentence in self]

    @property
    def heads(self):
        return [[0] + list(map(int, sentence.HEAD)) for sentence in self]

    @property
    def rels(self):
        return [[self.root] + list(sentence.DEPREL) for sentence in self]

    @property
    def pdeprels(self):
        return  [list(sentence.PDEPREL) for sentence in self]

    @property
    def tasks(self):
        return [sentence.TASK for sentence in self]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(HEAD=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(DEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @pdeprels.setter
    def pdeprels(self, sequences):
        self.sentences = [sentence._replace(PDEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname, task):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line.strip() for line in f]
        for i, line in enumerate(lines):
            if not line:
                sentence = Sentence(*zip(*[l.split() for l in lines[start:i]]), task)
                sentences.append(sentence)
                start = i + 1
        sentences = [sentence for sentence in sentences
                    if isprojective(list(map(int, sentence.HEAD)))]
        corpus = cls(sentences, task)

        return corpus

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self}\n")

    def __add__(self, another):
        assert isinstance(another, type(self))
        return Corpus(self.sentences + another.sentences)

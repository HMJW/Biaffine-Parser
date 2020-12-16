# -*- coding: utf-8 -*-

from biaffine_parser import BiaffineParser, Model
from biaffine_parser.utils import Corpus
from biaffine_parser.utils.data import TextDataset, batchify

import torch


class Hidden(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        return subparser

    def __call__(self, config):
        print("Load the model")
        vocab = torch.load(config.vocab)
        parser = BiaffineParser.load(config.model)
        model = Model(config, vocab, parser)

        words = ["I", "come", "from", "China"]

        feature = model.get_hidden(words)
        print(feature.shape)

# -*- coding: utf-8 -*-

from Dep import Dep, Model
from Dep.utils import Corpus
from Dep.utils.data import TextDataset, batchify
from shared import get_config
import torch


class Predict(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='path to predicted result')
        subparser.add_argument('--save_path', '-m', default='Dep/save/',
                               help='path to model file')
        return subparser

    def __call__(self, args):
        # reload parser
        dep = Dep.load(args.save_path)
        sentence = [["我", "是", "中国人"], ["你", "好"]]
        pos_list = [["PN", "VC", "NN"], ["PN", "VA"]]
        pred = dep.predict_batch(sentence, pos_list)
        print(pred) 

        # print("Load the model")
        # vocab = torch.load(config.vocab)
        # parser = BiaffineParser.load(config.model)
        # model = Model(vocab, parser)

        # print("Load the dataset")
        # corpus = Corpus.load(config.fdata)
        # dataset = TextDataset(vocab.numericalize(corpus, False))
        # # set the data loader
        # loader = batchify(dataset, config.batch_size)

        # print("Make predictions on the dataset")
        # corpus.heads, corpus.rels = model.predict(loader)

        # print(f"Save the predicted result to {config.fpred}")
        # corpus.save(config.fpred)

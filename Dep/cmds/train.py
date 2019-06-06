# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from Dep import Dep, Model
from Dep.metric import Metric
from Dep.utils import Corpus, Embedding, Vocab
from Dep.utils.data import TextDataset, batchify

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from shared import get_config

class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--ftrain', default='/data/wjiang/data/CoNLL09/train.auto.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='/data/wjiang/data/CoNLL09/dev.auto.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='/data/wjiang/data/CoNLL09/test.auto.conllx',
                               help='path to test file')
        subparser.add_argument('--fembed', default='/data/wjiang/data/embedding/giga.100.txt',
                               help='path to pretrained embeddings')
        subparser.add_argument('--unk', default=None,
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--save_path', '-m', default='Dep/save/',
                               help='path to model file')

        subparser.add_argument('--conf', '-c', default='config.json',
                               help='path to config file')
        return subparser

    def __call__(self, args):
        config = get_config(args.conf)

        print("Preprocess the data")
        train = Corpus.load(args.ftrain)
        dev = Corpus.load(args.fdev)
        test = Corpus.load(args.ftest)

        vocab_path = os.path.join(args.save_path, "vocab.pt")
        if os.path.exists(vocab_path):
            vocab = torch.load(vocab_path)
        else:
            vocab = Vocab.from_corpus(corpus=train, min_freq=2)
            vocab.read_embeddings(Embedding.load(args.fembed, args.unk))
            torch.save(vocab, vocab_path)
            
        print(vocab)

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(train))
        devset = TextDataset(vocab.numericalize(dev))
        testset = TextDataset(vocab.numericalize(test))
        # set the data loaders
        train_loader = batchify(dataset=trainset,
                                batch_size=config.batch_size,
                                shuffle=True)
        dev_loader = batchify(dataset=devset,
                              batch_size=config.batch_size)
        test_loader = batchify(dataset=testset,
                               batch_size=config.batch_size)
        print(f"{'train:':6} {len(trainset):5} sentences in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {len(devset):5} sentences in total, "
              f"{len(dev_loader):3} batches provided")
        print(f"{'test:':6} {len(testset):5} sentences in total, "
              f"{len(test_loader):3} batches provided")

        print("Create the model")
        parser = Dep(vocab, config, vocab.embeddings)
        if torch.cuda.is_available():
            parser = parser.cuda()
        print(f"{parser}\n")

        model = Model(vocab, parser)

        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        model.optimizer = Adam(model.parser.parameters(),
                               config.lr,
                               (config.beta_1, config.beta_2),
                               config.epsilon)
        model.scheduler = ExponentialLR(model.optimizer,
                                        config.decay ** (1 / config.steps))

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(train_loader)

            print(f"Epoch {epoch} / {config.epochs}:")
            loss, train_metric = model.evaluate(train_loader, args.punct)
            print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = model.evaluate(dev_loader, args.punct)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = model.evaluate(test_loader, args.punct)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                model.parser.save(args.save_path)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.parser = Dep.load(args.model)
        loss, metric = model.evaluate(test_loader, args.punct)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

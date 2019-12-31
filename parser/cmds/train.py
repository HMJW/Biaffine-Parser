# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from parser import BiaffineParser, Model
from parser.metric import Metric
from parser.utils import Corpus, Embedding, Vocab
from parser.utils.data import TextDataset, batchify
from parser.modules import BertEmbedding

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


def print_corpus(corpuses, names):
    for c, n in zip(corpuses, names):
        print(f"{n:35} has {len(c):6} sentences")
    print()

def evaluate(model, loaders, names, punct):
    assert len(loaders) == len(names)
    total_metric, total_loss = Metric(), 0
    uas, las = [], []
    for loader, name in zip(loaders, names):
        loss, metric = model.evaluate(loader, punct)
        total_metric += metric
        total_loss += loss
        uas.append(metric.uas)
        las.append(metric.las)
        print(f"{name:6} Loss: {loss:.4f} {metric}")
    print(f"{'mixed':8} {total_metric}")
    print(f"{'average':8} UAS: {sum(uas)/len(uas):.2%} LAS: {sum(las)/len(las):.2%}")
    return total_metric
    
class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--ftrain', default='../data/treebanks-filtered/codt/train.conll ../data/treebanks-filtered/ctb9/train.conll ../data/treebanks-filtered/hit/train.conll ../data/treebanks-filtered/pmt/train.conll',
                               help='path to train file')
        subparser.add_argument('--fdev', default='../data/treebanks-filtered/codt/dev.conll ../data/treebanks-filtered/ctb9/dev.conll ../data/treebanks-filtered/hit/dev.conll ../data/treebanks-filtered/pmt/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='../data/treebanks-filtered/codt/test.conll ../data/treebanks-filtered/ctb9/test.conll ../data/treebanks-filtered/hit/test.conll ../data/treebanks-filtered/pmt/test.conll' ,
                               help='path to test file')
        subparser.add_argument('--fembed', default='../data/embedding/giga.100.txt',
                               help='path to pretrained embeddings')
        subparser.add_argument('--unk', default=None,
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--task', default="codt ctb9 hit pmt",
                               help='all treebanks')
        subparser.add_argument('--tree', action='store_true',
                               help='whether to force tree')
        subparser.add_argument('--marg', action='store_true',
                               help='whether to use margin prob')
        subparser.add_argument('--crf', action='store_true',
                               help='whether to use crf loss')
        subparser.add_argument('--partial', action='store_true',
                               help='whether to partial')
        return subparser

    def __call__(self, config):
        print("Preprocess the data")
        task = config.task.split()
        ftrain = config.ftrain.split()
        fdev = config.fdev.split()
        ftest = config.ftest.split()
        assert len(ftrain) == len(fdev) == len(ftest) == len(task)

        trains = [Corpus.load(f, i) for i, f in enumerate(ftrain)]
        devs = [Corpus.load(f, i) for i, f in enumerate(fdev)]
        tests = [Corpus.load(f, i) for i, f in enumerate(ftest)]
        print_corpus(trains, ftrain)
        print_corpus(devs, fdev)
        print_corpus(tests, ftest)

        vocab_path = os.path.join(config.save_path, "vocab.pt")
        ext_emb_path = os.path.join(config.save_path, "ext_emb.pt")
        model_path = os.path.join(config.save_path, "model.pt")
        config_path = os.path.join(config.save_path, "config.pt")

        if config.preprocess or not os.path.exists(vocab_path):
            vocab = Vocab.from_corpus(corpus=trains, min_freq=2, task=task, bert_vocab=config.bert_model)
            emb = vocab.read_embeddings(Embedding.load(config.fembed, config.unk))
            torch.save(vocab, vocab_path)
            torch.save(emb, ext_emb_path)
        else:
            vocab = torch.load(vocab_path)
            emb = torch.load(config_path)
        config.update({
            'n_words': vocab.n_init,
            'n_chars': vocab.n_chars,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        torch.save(config, config_path)

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(
            sum(trains, Corpus([]))), config.buckets)
        devsets = [TextDataset(vocab.numericalize(dev), config.buckets) for dev in devs]
        testsets = [TextDataset(vocab.numericalize(test), config.buckets) for test in tests]

        # set the data loaders
        train_loader = batchify(trainset, config.batch_size, True)
        dev_loaders = [batchify(devset, config.batch_size) for devset in devsets]
        test_loaders = [batchify(testset, config.batch_size) for testset in testsets]
        print(f"{'train:':6} {len(trainset):5} sentences in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {sum(len(devset) for devset in devsets):5} sentences in total, "
              f"{sum(len(dev_loader) for dev_loader in dev_loaders):3} batches provided")
        print(f"{'test:':6} {sum(len(testset) for testset in testsets):5} sentences in total, "
              f"{sum(len(test_loader) for test_loader in test_loaders):3} batches provided")

        print("Create the model")
        parser = BiaffineParser(config, vocab, emb).to(config.device)
        bert = BertEmbedding(config.bert_model, config.bert_layer)
        print(f"{parser}\n")

        model = Model(config, vocab, parser, bert)

        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        model.optimizer = Adam(model.parser.parameters(),
                               config.lr,
                               (config.mu, config.nu),
                               config.epsilon)
        model.scheduler = ExponentialLR(model.optimizer,
                                        config.decay**(1/config.decay_steps))

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(train_loader)

            print(f"Epoch {epoch} / {config.epochs}:")
            # loss, train_metric = model.evaluate(train_loader, config.punct)
            # print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")

            print(f"{'dev:':6}")
            dev_metric = evaluate(model, dev_loaders, task, config.punct)
            print(f"{'test:':6}")
            evaluate(model, test_loaders, task, config.punct)

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                model.parser.save(model_path)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.parser = BiaffineParser.load(config.save_path)
        metric = evaluate(model, test_loaders, task, config.punct)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

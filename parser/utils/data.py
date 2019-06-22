# -*- coding: utf-8 -*-

from collections.abc import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


class TextDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(TextDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        for raw_batch in super(TextDataLoader, self).__iter__():
            batch, device = [], 'cuda' if torch.cuda.is_available() else 'cpu'
            for data in raw_batch:
                if isinstance(data[0], torch.Tensor):
                    data = pad_sequence(data, True).to(device)
                elif isinstance(data[0], Iterable):
                    data = [pad_sequence(f, True).to(device)
                            for f in zip(*data)]
                batch.append(data)
            yield batch


class TextDataset(Dataset):

    def __init__(self, items, n_buckets=1):
        super(TextDataset, self).__init__()

        self.items = items
        # NOTE: the final bucket count is less than or equal to n_buckets
        self.centroids, self.clusters = kmeans(x=[len(i) for i in items[-1]],
                                               k=n_buckets)
        self.buckets = dict(zip(self.centroids, self.clusters))

    def __getitem__(self, index):
        return tuple(item[index] for item in self.items)

    def __len__(self):
        return len(self.items[0])

    @classmethod
    def collate_fn(cls, batch):
        return (field for field in zip(*batch))


class TextSampler(Sampler):

    def __init__(self, buckets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[
            (size, bucket) for size, bucket in buckets.items()
        ])
        # number of chunks in each bucket
        self.chunks = [max(round(size * len(bucket) / batch_size), 1)
                       for size, bucket in zip(self.sizes, self.buckets)]

    def __iter__(self):
        # if shuffle, shffule both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                           for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        return sum(self.chunks)


def batchify(dataset, batch_size, shuffle=False):
    batch_sampler = TextSampler(buckets=dataset.buckets,
                                batch_size=batch_size,
                                shuffle=shuffle)
    loader = TextDataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=TextDataset.collate_fn)

    return loader


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # initialize k centroids randomly
    c, old = x[torch.randperm(len(x))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        # update the centroids
        c, old = torch.tensor([x[y.eq(i)].mean() for i in range(k)]), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)
    clusters = [y.eq(i) for i in range(k)]
    clusters = [i.nonzero().view(-1).tolist() for i in clusters if i.any()]
    centroids = [round(x[i].mean().item()) for i in clusters]

    return centroids, clusters

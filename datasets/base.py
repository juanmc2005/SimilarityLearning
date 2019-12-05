#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import torch


class SimDatasetPartition:

    def __iter__(self):
        return self

    def nbatches(self) -> int:
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class SimDataset:

    def training_partition(self) -> SimDatasetPartition:
        raise NotImplementedError

    def dev_partition(self) -> SimDatasetPartition:
        raise NotImplementedError

    def test_partition(self) -> SimDatasetPartition:
        raise NotImplementedError


class DataSplitter:

    def split(self, data):
        raise NotImplementedError


class LoaderWrapperPartition(SimDatasetPartition):

    def __init__(self, loader):
        super(LoaderWrapperPartition, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def nbatches(self) -> int:
        return len(self.loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class TextPartition(SimDatasetPartition):

    def __init__(self, data, batch_size: int, train: bool, batches_per_epoch: int = None):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.batches_per_epoch = batches_per_epoch
        self.generator = self._generate()

    def _transform_batch(self, x, y):
        raise NotImplementedError("The class should implement 'transform_batch'")

    def _generate(self):
        while True:
            if self.train:
                np.random.shuffle(self.data)
            for i in range(0, len(self.data), self.batch_size):
                j = min(i + self.batch_size, len(self.data))
                yield self.data[i:j]

    def nbatches(self):
        total_batches = math.ceil(len(self.data) / self.batch_size)
        if self.batches_per_epoch is not None and self.batches_per_epoch in range(1, total_batches):
            return self.batches_per_epoch
        else:
            return total_batches

    def __next__(self):
        batch = next(self.generator)
        x, y = [x for x, _ in batch], torch.Tensor([y for _, y in batch])
        return self._transform_batch(x, y)


class ClassBalancedTextPartition(TextPartition):

    def __init__(self, data, per_class: int, nclass: int, batches_per_epoch: int = None):
        super(ClassBalancedTextPartition, self).__init__(data, per_class * nclass, True, batches_per_epoch)
        self.per_class = per_class
        self.nclass = nclass

    def _generate(self):
        x, y = [x for x, _ in self.data], [y for _, y in self.data]
        while True:
            remaining_x, remaining_y = x, y
            while len(remaining_y) > self.batch_size:
                counters = [self.per_class for _ in range(self.nclass)]
                perm = np.random.permutation(len(remaining_y))
                new_remaining_x, new_remaining_y, batch = [], [], []
                for i in perm:
                    if counters[remaining_y[i]] > 0:
                        counters[remaining_y[i]] -= 1
                        batch.append((remaining_x[i], remaining_y[i]))
                    else:
                        new_remaining_x.append(remaining_x[i])
                        new_remaining_y.append(remaining_y[i])
                remaining_x, remaining_y = new_remaining_x, new_remaining_y
                yield batch
            yield list(zip(remaining_x, remaining_y))


class TextLongLabelPartition(TextPartition):

    def _transform_batch(self, x, y):
        return x, y.long()


class ClassBalancedTextLongLabelPartition(ClassBalancedTextPartition):

    def _transform_batch(self, x, y):
        return x, y.long()


class TextFloatLabelPartition(TextPartition):

    def _transform_batch(self, x, y):
        return x, y.float()

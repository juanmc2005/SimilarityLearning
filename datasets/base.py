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

    def __init__(self, data, batch_size: int, train: bool):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.generator = self._generate()

    def _transform_batch(self, x, y):
        raise NotImplementedError("The class should implement 'transform_batch'")

    def _generate(self):
        while True:
            if self.train:
                np.random.shuffle(self.data)
                print(self.data)
            for i in range(0, len(self.data), self.batch_size):
                j = min(i + self.batch_size, len(self.data))
                yield self.data[i:j]

    def nbatches(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __next__(self):
        batch = next(self.generator)
        x, y = [x for x, _ in batch], torch.Tensor([y for _, y in batch])
        return self._transform_batch(x, y)


class TextLongLabelPartition(TextPartition):

    def _transform_batch(self, x, y):
        return x, y.long()


class TextFloatLabelPartition(TextPartition):

    def _transform_batch(self, x, y):
        return x, y.float()

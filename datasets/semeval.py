#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from os.path import join

import numpy as np
import torch

from datasets.base import SimDataset, SimDatasetPartition
from sts.augmentation import SemEvalAugmentationStrategy, pad_sent_pair
from sts import utils as sts


class SemEvalPartition(SimDatasetPartition):

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
            for i in range(0, len(self.data), self.batch_size):
                j = min(i + self.batch_size, len(self.data))
                yield self.data[i:j]

    def nbatches(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __next__(self):
        batch = next(self.generator)
        x, y = [x for x, _ in batch], torch.Tensor([y for _, y in batch])
        return self._transform_batch(x, y)


class SemEvalBaselinePartition(SemEvalPartition):

    def _transform_batch(self, x, y):
        y = y.float()
        y = y.view(-1, 6) if self.train else y
        return x, y


class SemEvalLongLabelPartition(SemEvalPartition):

    def _transform_batch(self, x, y):
        return x, y.long()


class SemEvalFloatLabelPartition(SemEvalPartition):

    def _transform_batch(self, x, y):
        return x, y.float()


class SemEvalPartitionFactory:

    def __init__(self, loss: str, batch_size: int):
        self.loss = loss
        self.batch_size = batch_size

    def new(self, data, train: bool):
        if self.loss == 'kldiv':
            return SemEvalBaselinePartition(data, self.batch_size, train)
        elif self.loss == 'contrastive' or self.loss == 'triplet':
            return SemEvalFloatLabelPartition(data, self.batch_size, train)
        else:
            # Softmax based loss, classes are simulated with clusters
            return SemEvalLongLabelPartition(data, self.batch_size, train)


class SemEval(SimDataset):

    def __init__(self, path: str, vector_path: str, vocab_path: str,
                 augmentation: SemEvalAugmentationStrategy, partition_factory: SemEvalPartitionFactory):
        self.path = path
        self.augmentation = augmentation
        self.partition_factory = partition_factory
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        atrain, btrain, simtrain = self._load_partition('train')
        adev, bdev, simdev = self._load_partition('dev')
        atest, btest, simtest = self._load_partition('test')
        self.train_sents = self.augmentation.augment(atrain, btrain, simtrain)
        self.dev_sents = np.array(list(zip(self._split_and_pad(adev, bdev), simdev)))
        self.test_sents = np.array(list(zip(self._split_and_pad(atest, btest), simtest)))

    def _split_and_pad(self, asents, bsents):
        result = []
        for a, b in zip(asents, bsents):
            apad, bpad = pad_sent_pair(a.split(' '), b.split(' '))
            result.append((apad, bpad))
        return result

    def _load_partition(self, partition):
        with open(join(self.path, partition, 'a.toks')) as afile, \
                open(join(self.path, partition, 'b.toks')) as bfile, \
                open(join(self.path, partition, 'sim.txt')) as simfile:
            a = [line.strip() for line in afile.readlines()]
            b = [line.strip() for line in bfile.readlines()]
            sim = [float(line.strip()) for line in simfile.readlines()]
            return a, b, sim

    @property
    def nclass(self):
        return self.augmentation.nclass()

    def training_partition(self) -> SimDatasetPartition:
        np.random.shuffle(self.train_sents)
        return self.partition_factory.new(self.train_sents, train=True)

    def dev_partition(self) -> SimDatasetPartition:
        return self.partition_factory.new(self.dev_sents, train=False)

    def test_partition(self) -> SimDatasetPartition:
        return self.partition_factory.new(self.test_sents, train=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join
import numpy as np
import datasets.base as base
from sts.augmentation import SemEvalAugmentationStrategy, pad_sent_pair
from sts import utils as sts


class SemEvalBaselinePartition(base.TextPartition):

    def _transform_batch(self, x, y):
        y = y.float()
        y = y.view(-1, 6) if self.train else y
        return x, y


class SemEvalPartitionFactory:

    def __init__(self, loss: str, batch_size: int):
        self.loss = loss
        self.batch_size = batch_size

    def new(self, data, train: bool):
        if self.loss == 'kldiv':
            return SemEvalBaselinePartition(data, self.batch_size, train)
        elif self.loss == 'contrastive' or self.loss == 'triplet':
            return base.TextFloatLabelPartition(data, self.batch_size, train)
        else:
            # Softmax based loss, classes are simulated with clusters
            return base.TextLongLabelPartition(data, self.batch_size, train)


class SemEval(base.SimDataset):

    def __init__(self, path: str, vector_path: str, vocab_path: str,
                 augmentation: SemEvalAugmentationStrategy, partition_factory: SemEvalPartitionFactory):
        self.path = path
        self.augmentation = augmentation
        self.partition_factory = partition_factory
        if vector_path is not None:
            self.vocab_vec, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
            self.vocab = list(self.vocab_vec.keys())
            print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        else:
            self.vocab_vec = None
            self.vocab = [line.strip() for line in open(vocab_path, 'r')]
        atrain, btrain, simtrain = self._load_partition('train')
        adev, bdev, simdev = self._load_partition('dev')
        atest, btest, simtest = self._load_partition('test')
        self.train_sents = self.augmentation.augment(atrain, btrain, simtrain)
        self.dev_sents = np.array(list(zip(self._split_and_pad(adev, bdev), simdev)))
        self.test_sents = np.array(list(zip(self._split_and_pad(atest, btest), simtest)))
        print(f"Unique Dev Sentences: {len(set(adev + bdev))}")
        print(f"Unique Test Sentences: {len(set(atest + btest))}")

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

    def training_partition(self) -> base.SimDatasetPartition:
        np.random.shuffle(self.train_sents)
        return self.partition_factory.new(self.train_sents, train=True)

    def dev_partition(self) -> base.SimDatasetPartition:
        return self.partition_factory.new(self.dev_sents, train=False)

    def test_partition(self) -> base.SimDatasetPartition:
        return self.partition_factory.new(self.test_sents, train=False)

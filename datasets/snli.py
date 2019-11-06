#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join
import numpy as np
import datasets.base as base
from sts.augmentation import pad_sent_pair, SemEvalAugmentationStrategy
from sts import utils as sts
from datasets.base import TextFloatLabelPartition


class SNLI(base.SimDataset):

    def __init__(self, path: str, vector_path: str, vocab_path: str, batch_size: int,
                 augmentation: SemEvalAugmentationStrategy, label2int: dict):
        self.path = path
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.label2int = label2int
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        atrain, btrain, simtrain = self._load_partition('train')
        adev, bdev, simdev = self._load_partition('dev')
        atest, btest, simtest = self._load_partition('test')
        self.train_sents = self.augmentation.augment(atrain, btrain, simtrain)
        self.dev_sents = np.array(list(zip(self._split_and_pad(adev, bdev), simdev)))
        self.test_sents = np.array(list(zip(self._split_and_pad(atest, btest), simtest)))
        print(f"Unique Train Sentences: {len(set(atrain + btrain))}")
        print(f"Unique Dev Sentences: {len(set(adev + bdev))}")
        print(f"Unique Test Sentences: {len(set(atest + btest))}")

    def _split_and_pad(self, asents, bsents):
        result = []
        for a, b in zip(asents, bsents):
            apad, bpad = pad_sent_pair(a.split(' '), b.split(' '))
            result.append((apad, bpad))
        return result

    def _load_partition(self, partition):
        with open(join(self.path, partition, 's1')) as afile, \
                open(join(self.path, partition, 's2')) as bfile, \
                open(join(self.path, partition, 'labels')) as simfile:
            a = [f"<s> {line.strip()} </s>" for line in afile.readlines()]
            b = [f"<s> {line.strip()} </s>" for line in bfile.readlines()]
            sim = [self.label2int[line.strip()] for line in simfile.readlines()]
            return a, b, sim

    @property
    def nclass(self):
        return self.augmentation.nclass()

    # Using float labels because only contrastive and triplet loss are supported

    def training_partition(self) -> base.SimDatasetPartition:
        np.random.shuffle(self.train_sents)
        return TextFloatLabelPartition(self.train_sents, self.batch_size, train=True)

    def dev_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.dev_sents, self.batch_size, train=False)

    def test_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.test_sents, self.batch_size, train=False)

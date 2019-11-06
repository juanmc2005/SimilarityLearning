#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join
import numpy as np
import datasets.base as base
from sts.augmentation import pad_sent_pair
from sts import utils as sts
from datasets.base import TextFloatLabelPartition


class SNLI(base.SimDataset):

    def __init__(self, path: str, vector_path: str, vocab_path: str, batch_size: int):
        self.label2int = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
        self.path = path
        self.batch_size = batch_size
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        atrain, btrain, simtrain = self._load_partition('train')
        atrain, btrain, simtrain = self._treat_neutrals(atrain, btrain, simtrain)
        adev, bdev, simdev = self._load_partition('dev')
        atest, btest, simtest = self._load_partition('test')
        self.train_sents = np.array(list(zip(self._split_and_pad(atrain, btrain), simtrain)))
        self.dev_sents = np.array(list(zip(self._split_and_pad(adev, bdev), simdev)))
        self.test_sents = np.array(list(zip(self._split_and_pad(atest, btest), simtest)))
        print(f"Unique Train Sentences: {len(set(atrain + btrain))}")
        print(f"Unique Dev Sentences: {len(set(adev + bdev))}")
        print(f"Unique Test Sentences: {len(set(atest + btest))}")
        print(f"Train sents: \n{self.train_sents[:3]}")
        print(f"Dev sents: \n{self.dev_sents[:3]}")
        print(f"Test sents: \n{self.test_sents[:3]}")

    def _split_and_pad(self, asents, bsents):
        result = []
        for a, b in zip(asents, bsents):
            apad, bpad = pad_sent_pair(a.split(' '), b.split(' '))
            result.append((apad, bpad))
        return result

    def _treat_neutrals(self, asents, bsents, labels):
        id_neutral = self.label2int['neutral']
        a = [sent for i, sent in enumerate(asents) if labels[i] != id_neutral]
        b = [sent for i, sent in enumerate(bsents) if labels[i] != id_neutral]
        sim = [1 if s == self.label2int['contradiction'] else 0 for s in labels if s != id_neutral]
        return a, b, sim

    def _load_partition(self, partition):
        with open(join(self.path, partition, 's1')) as afile, \
                open(join(self.path, partition, 's2')) as bfile, \
                open(join(self.path, partition, 'labels')) as simfile:
            a = [f"<s> {line.strip()} </s>" for line in afile.readlines()]
            b = [f"<s> {line.strip()} </s>" for line in bfile.readlines()]
            sim = [self.label2int[line.strip()] for line in simfile.readlines()]
            return a, b, sim

    # Using float labels because only contrastive and triplet loss are supported

    def training_partition(self) -> base.SimDatasetPartition:
        np.random.shuffle(self.train_sents)
        return TextFloatLabelPartition(self.train_sents, self.batch_size, train=True)

    def dev_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.dev_sents, self.batch_size, train=False)

    def test_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.test_sents, self.batch_size, train=False)

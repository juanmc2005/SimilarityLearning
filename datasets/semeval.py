#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
from os.path import join
import sts_utils as sts
from datasets.base import SimDataset, SimDatasetPartition


class SemEvalPartition(SimDatasetPartition):

    def __init__(self, data, batch_size: int, train: bool):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.generator = self._generate()

    def _transform_batch(self, batch):
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
        return self._transform_batch(next(self.generator))


class SemEvalBaselinePartition(SemEvalPartition):

    def _transform_batch(self, batch):
        x = [x for x, _ in batch]
        y = torch.Tensor([y for _, y in batch]).float()
        y = y.view(-1, 6) if self.train else y
        return x, y


class SemEvalClusterizedPartition(SemEvalPartition):

    def _transform_batch(self, batch):
        return [x for x, _ in batch], torch.Tensor([y for _, y in batch]).long()


class SemEvalContrastivePartition(SemEvalPartition):

    def _transform_batch(self, batch):
        return [x for x, _ in batch], torch.Tensor([y for _, y in batch]).float()


class SemEvalPartitionBuilder:

    def __init__(self, batch_size: int, mode: str):
        self.batch_size = batch_size
        self.mode = mode

    def build(self, data, train: bool):
        # TODO add 'triplets' mode
        if self.mode == 'baseline':
            return SemEvalBaselinePartition(data, self.batch_size, train)
        elif self.mode == 'pairs':
            return SemEvalContrastivePartition(data, self.batch_size, train)
        elif self.mode == 'clusters':
            return SemEvalClusterizedPartition(data, self.batch_size, train)
        else:
            raise ValueError("Mode can only be 'baseline', 'clusters', 'pairs' or 'triplets'")


class SemEval(SimDataset):

    @staticmethod
    def pad_sent(s1, s2):
        if len(s1) == len(s2):
            return s1, s2
        elif len(s1) > len(s2):
            d = len(s2)
            for i in range(d, len(s1)):
                s2 += ' null'
        else:
            d = len(s1)
            for i in range(d, len(s2)):
                s1 += 'null'
        return s1, s2

    @staticmethod
    def scores_to_probs(scores):
        labels = []
        for s in scores:
            ceil = int(math.ceil(s))
            floor = int(math.floor(s))
            tmp = [0, 0, 0, 0, 0, 0]
            if floor != ceil:
                tmp[ceil] = s - floor
                tmp[floor] = ceil - s
            else:
                tmp[floor] = 1
            labels.append(tmp)
        return labels

    def __init__(self, path, vector_path, vocab_path, batch_size, mode='baseline', threshold=2.5, allow_redundancy=True):
        # TODO mode parameter should be refactored into a strategy-like object
        self.path = path
        self.batch_size = batch_size
        self.mode = mode
        self.allow_redundancy = allow_redundancy
        self.nclass = None
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {int(100 * n_inv / (n_inv + n_oov))}% coverage")
        atrain, btrain, simtrain = self._load_partition('train')
        adev, bdev, simdev = self._load_partition('dev')
        atest, btest, simtest = self._load_partition('test')
        self.dev_sents = np.array(list(zip(zip(adev, bdev), simdev)))
        self.test_sents = np.array(list(zip(zip(atest, btest), simtest)))
        if mode == 'clusters':
            sents_a = atrain + adev + atest
            sents_b = btrain + bdev + btest
            scores = simtrain + simdev + simtest
            sents_a, sents_b, scores = sts.unique_pairs(sents_a, sents_b, scores)
            clusters, self.nclass = self._clusterize(sents_a, sents_b, scores, threshold)
            self.train_sents, train_sents_raw, dev_sents, test_sents = [], [], [], []
            for i, cluster in enumerate(clusters):
                for sent in cluster:
                    if sent in atrain or sent in btrain:
                        self.train_sents.append((sent.split(' '), i))
                        train_sents_raw.append(sent)
                    if sent in adev or sent in bdev:
                        dev_sents.append(sent)
                    if sent in atest or sent in btest:
                        test_sents.append(sent)
            self.train_sents = np.array(self.train_sents)
            print(f"Unique sentences used for clustering: {len(set(sents_a + sents_b))}")
            print(f"Total Train Sentences: {len(set(atrain + btrain))}")
            print(f"Train Sentences Kept: {len(set(train_sents_raw))}")
            print(f"Total Dev Sentences: {len(set(adev + bdev))}")
            print(f"Dev Sentences Kept: {len(set(dev_sents))}")
            print(f"Total Test Sentences: {len(set(atest + btest))}")
            print(f"Test Sentences Kept: {len(set(test_sents))}")
            print(f"N Clusters: {self.nclass}")
            print(f"Max Cluster Size: {max([len(cluster) for cluster in clusters])}")
            print(f"Mean Cluster Size: {np.mean([len(cluster) for cluster in clusters])}")
        elif mode == 'pairs':
            self.train_sents = self._pairs(atrain, btrain, simtrain, threshold)
            print(f"Original Train Pairs: {len(atrain)}")
            print(f"Original Unique Train Pairs: {len(set(zip(atrain, btrain)))}")
            print(f"Total Train Pairs: {len(self.train_sents)}")
            print(f"+ Train Pairs: {len([y for _, y in self.train_sents if y == 0])}")
            print(f"- Train Pairs: {len([y for _, y in self.train_sents if y == 1])}")
        elif mode == 'triplets':
            self.train_sents = self._triplets(atrain, btrain, simtrain, threshold)
        elif mode == 'baseline':
            if self.allow_redundancy:
                sim = self.scores_to_probs(simtrain)
                self.train_sents = np.array(list(zip(zip(atrain, btrain), sim)))
                print(f"Train Pairs: {len(atrain)}")
                print("Redundancy in the training set is allowed")
            else:
                unique_train_data = list(set(zip(atrain, btrain, simtrain)))
                pairs = [(x1, x2) for x1, x2, _ in unique_train_data]
                sim = self.scores_to_probs([y for _, _, y in unique_train_data])
                self.train_sents = np.array(list(zip(pairs, sim)))
                print(f"Original Train Pairs: {len(atrain)}")
                print(f"Unique Train Pairs: {len(unique_train_data)}")
        else:
            raise ValueError("Mode can only be 'baseline', 'clusters', 'pairs' or 'triplets'")

    def training_partition(self) -> SimDatasetPartition:
        np.random.shuffle(self.train_sents)
        return SemEvalPartitionBuilder(self.batch_size, self.mode).build(self.train_sents, train=True)

    def dev_partition(self) -> SimDatasetPartition:
        np.random.shuffle(self.dev_sents)
        return SemEvalPartitionBuilder(self.batch_size, self.mode).build(self.dev_sents, train=False)

    def test_partition(self) -> SimDatasetPartition:
        np.random.shuffle(self.test_sents)
        return SemEvalPartitionBuilder(self.batch_size, self.mode).build(self.test_sents, train=False)

    def _load_partition(self, partition):
        with open(join(self.path, partition, 'a.toks')) as afile, \
                open(join(self.path, partition, 'b.toks')) as bfile, \
                open(join(self.path, partition, 'sim.txt')) as simfile:
            a = [line.strip() for line in afile.readlines()]
            b = [line.strip() for line in bfile.readlines()]
            sim = [float(line.strip()) for line in simfile.readlines()]
            for i in range(len(a)):
                a[i], b[i] = self.pad_sent(a[i], b[i])
            return a, b, sim

    def _clusterize(self, sents_a, sents_b, scores, threshold):
        segment_a = sts.SemEvalSegment(sents_a)
        segment_b = sts.SemEvalSegment(sents_b)
        clusters = segment_a.clusters(segment_b, scores, threshold)
        return clusters, len(clusters)

    def _pairs(self, sents_a, sents_b, scores, threshold):
        segment_a = sts.SemEvalSegment(sents_a)
        segment_b = sts.SemEvalSegment(sents_b)
        pos, neg = sts.pairs(segment_a, segment_b, scores, threshold)
        data = [((s1, s2), 0) for s1, s2 in pos] + [((s1, s2), 1) for s1, s2 in neg]
        return np.array([((s1.split(' '), s2.split(' ')), y) for (s1, s2), y in data])

    def _triplets(self, sents_a, sents_b, scores, threshold):
        segment_a = sts.SemEvalSegment(sents_a)
        segment_b = sts.SemEvalSegment(sents_b)
        unique_sents = set(sents_a + sents_b)
        pos, neg = sts.pairs(segment_a, segment_b, scores, threshold)
        anchors, positives, negatives = sts.triplets(unique_sents, pos, neg)
        return np.array([(a.split(' '), p.split(' '), n.split(' '))
                         for a, p, n in zip(anchors, positives, negatives)])
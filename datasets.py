#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from pyannote.audio.features.utils import RawAudio
# from pyannote.audio.embedding.generators import SpeechSegmentGenerator
# from pyannote.database import get_protocol
# from pyannote.database import FileFinder
from os.path import join
import sts_utils as sts


class SimDataset:

    def training_partition(self):
        raise NotImplementedError

    def test_partition(self):
        raise NotImplementedError


class SimDatasetPartition:

    def nbatches(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class LoaderWrapperPartition(SimDatasetPartition):

    def __init__(self, loader):
        super(LoaderWrapperPartition, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def nbatches(self):
        return len(self.loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class MNIST(SimDataset):

    def __init__(self, path, batch_size):
        super(MNIST, self).__init__()
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        self.trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
        self.testset = datasets.MNIST(path, download=True, train=False, transform=transform)

    def training_partition(self):
        return LoaderWrapperPartition(DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=4))

    def test_partition(self):
        return LoaderWrapperPartition(DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=4))


class VoxCelebPartition(SimDatasetPartition):

    def __init__(self, generator):
        self.generator = generator()
        self.batches_per_epoch = generator.batches_per_epoch
        self.batch_size = generator.batch_size

    def nbatches(self):
        return self.batches_per_epoch

    def __next__(self):
        dic = next(self.generator)
        return torch.Tensor(dic['X']).view(self.batch_size, -1), torch.Tensor(dic['y']).long()


class VoxCeleb1(SimDataset):

    def __init__(self, batch_size):
        extractor = RawAudio(sample_rate=16000)
        preprocessors = {'audio': FileFinder()}
        protocol = get_protocol('VoxCeleb.SpeakerVerification.VoxCeleb1_X', preprocessors=preprocessors)
        self.train_gen = SpeechSegmentGenerator(extractor, protocol, subset='train', per_label=1,
                                                per_fold=batch_size, duration=0.2, parallel=3)
        self.dev_gen = SpeechSegmentGenerator(extractor, protocol, subset='development', per_label=1,
                                              per_fold=batch_size, duration=0.2, parallel=2)

    def training_partition(self):
        return VoxCelebPartition(self.train_gen)

    def test_partition(self):
        return VoxCelebPartition(self.dev_gen)


class SemEvalClusterizedPartition(SimDatasetPartition):

    def __init__(self, sent_data, batch_size):
        self.sents = [x for x, _ in sent_data]
        self.y = [y for _, y in sent_data]
        self.batch_size = batch_size
        self.generator = self._generate()

    def _generate(self):
        start = -self.batch_size
        while True:
            start += self.batch_size
            end = min(start + self.batch_size, len(self.sents))
            yield (self.sents[start:end], self.y[start:end])

    def nbatches(self):
        return math.ceil(len(self.sents) / self.batch_size)

    def __next__(self):
        return next(self.generator)


class SemEval(SimDataset):

    def __init__(self, path, vector_path, vocab_path, batch_size, mode='auto', threshold=2.5):
        # TODO mode parameter should be refactored into a strategy-like object
        self.path = path
        self.batch_size = batch_size
        self.nclass = None
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {int(100 * n_inv / (n_inv + n_oov))}% coverage")
        atrain, btrain, simtrain = self._load_partition('train')
        adev, bdev, simdev = self._load_partition('dev')
        atest, btest, simtest = self._load_partition('test')
        if mode == 'auto':
            sents_a = atrain + adev + atest
            sents_b = btrain + bdev + btest
            scores = simtrain + simdev + simtest
            sents_a, sents_b, scores = sts.unique_pairs(sents_a, sents_b, scores)
            clusters, self.nclass = self._clusterize(sents_a, sents_b, scores, threshold)
            self.train_sents, self.dev_sents = [], []
            for i, cluster in enumerate(clusters):
                for sent in cluster:
                    if sent in atrain or sent in btrain:
                        self.train_sents.append((sent.split(' '), i))
                    if sent in adev or sent in bdev:
                        self.dev_sents.append((sent.split(' '), i))
        elif mode == 'pairs':
            self.train_sents = self._pairs(atrain, btrain, simtrain, threshold)
            self.dev_sents = self._pairs(adev, bdev, simdev, threshold)
        elif mode == 'triplets':
            self.train_sents = self._triplets(atrain, btrain, simtrain, threshold)
            self.dev_sents = self._triplets(adev, bdev, simdev, threshold)
        else:
            raise ValueError("Mode can only be 'auto', 'pairs' or 'triplets'")

    def training_partition(self):
        np.random.shuffle(self.train_sents)
        return SemEvalClusterizedPartition(self.train_sents, self.batch_size)

    def test_partition(self):
        np.random.shuffle(self.dev_sents)
        return SemEvalClusterizedPartition(self.dev_sents, self.batch_size)

    def _load_partition(self, partition):
        with open(join(self.path, partition, 'a.toks')) as afile, \
                open(join(self.path, partition, 'b.toks')) as bfile, \
                open(join(self.path, partition, 'sim.txt')) as simfile:
            a = [line.strip() for line in afile.readlines()]
            b = [line.strip() for line in bfile.readlines()]
            sim = [float(line.strip()) for line in simfile.readlines()]
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
        data = [(s1, s2, 0) for s1, s2 in pos] + [(s1, s2, 1) for s1, s2 in neg]
        return np.array([(s1.split(' '), s2.split(' '), y) for s1, s2, y in data])

    def _triplets(self, sents_a, sents_b, scores, threshold):
        segment_a = sts.SemEvalSegment(sents_a)
        segment_b = sts.SemEvalSegment(sents_b)
        unique_sents = set(sents_a + sents_b)
        pos, neg = sts.pairs(segment_a, segment_b, scores, threshold)
        anchors, positives, negatives = sts.triplets(unique_sents, pos, neg)
        return np.array([(a.split(' '), p.split(' '), n.split(' '))
                         for a, p, n in zip(anchors, positives, negatives)])


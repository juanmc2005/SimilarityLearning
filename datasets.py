#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from os.path import join
import sts_utils as sts
from metrics import SpeakerValidationConfig


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

    def __init__(self, generator, segment_size):
        self.generator = generator()
        self.batches_per_epoch = generator.batches_per_epoch
        self.segment_size = segment_size

    def nbatches(self):
        return self.batches_per_epoch

    def __next__(self):
        dic = next(self.generator)
        x, y = torch.Tensor(dic['X']).view(-1, self.segment_size), torch.Tensor(dic['y']).long()
        return x, y


class VoxCeleb1(SimDataset):

    def __init__(self, batch_size, segment_size_millis):
        sample_rate = 16000
        segment_size_s = segment_size_millis / 1000
        print(f"Segment Size = {segment_size_s}s")
        self.nfeat = sample_rate * segment_size_millis // 1000
        print(f"Embedding Size = {self.nfeat}")
        self.config = SpeakerValidationConfig(protocol_name='VoxCeleb.SpeakerVerification.VoxCeleb1_X',
                                              feature_extraction=RawAudio(sample_rate=sample_rate),
                                              preprocessors={'audio': FileFinder()},
                                              duration=segment_size_s)
        protocol = get_protocol(self.config.protocol_name, preprocessors=self.config.preprocessors)
        self.train_gen = SpeechSegmentGenerator(self.config.feature_extraction, protocol,
                                                subset='train', per_label=1, per_fold=batch_size,
                                                duration=segment_size_s, parallel=3)
        self.dev_gen = SpeechSegmentGenerator(self.config.feature_extraction, protocol,
                                              subset='development', per_label=1, per_fold=batch_size,
                                              duration=segment_size_s, parallel=2)

    def training_partition(self):
        return VoxCelebPartition(self.train_gen, self.nfeat)

    def test_partition(self):
        return VoxCelebPartition(self.dev_gen, self.nfeat)


class SemEvalClusterizedPartition(SimDatasetPartition):

    def __init__(self, sent_data, batch_size):
        self.data = sent_data
        self.batch_size = batch_size
        self.generator = self._generate()

    def _generate(self):
        start = 0
        while True:
            end = min(start + self.batch_size, len(self.data))
            batch = self.data[start:end]
            if end == len(self.data):
                start = 0
            else:
                start += self.batch_size
            yield batch

    def nbatches(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __next__(self):
        batch = next(self.generator)
        np.random.shuffle(batch)
        return [x for x, _ in batch], torch.Tensor([y for _, y in batch]).long()


class SemEvalPairwisePartition(SimDatasetPartition):

    def __init__(self, data, batch_size, classes):
        self.data = data
        self.batch_size = batch_size
        self.classes = classes
        self.generator = self._generate()

    def _generate(self):
        start = 0
        while True:
            end = min(start + self.batch_size, len(self.data))
            batch = self.data[start:end]
            if end == len(self.data):
                start = 0
            else:
                start += self.batch_size
            yield batch

    def nbatches(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __next__(self):
        batch = next(self.generator)
        np.random.shuffle(batch)
        x = [x for x, _ in batch]
        y = torch.Tensor([y for _, y in batch]).float()
        y = y.view(-1, 6) if self.classes else y
        return x, y


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

    def __init__(self, path, vector_path, vocab_path, batch_size, mode='classic', threshold=2.5):
        # TODO mode parameter should be refactored into a strategy-like object
        self.path = path
        self.batch_size = batch_size
        self.mode = mode
        self.nclass = None
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {int(100 * n_inv / (n_inv + n_oov))}% coverage")
        atrain, btrain, simtrain = self._load_partition('train')
        adev, bdev, simdev = self._load_partition('dev')
        atest, btest, simtest = self._load_partition('test')
        if mode == 'clusters':
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
            self.train_sents = np.array(self.train_sents)
            self.dev_sents = np.array(self.dev_sents)
            print(f"Unique sentences used for clustering: {len(set(sents_a + sents_b))}")
            print(f"Train Sentences: {len(self.train_sents)}")
            print(f"Dev Sentences: {len(self.dev_sents)}")
            print(f"N Clusters: {self.nclass}")
            print(f"Max Cluster Size: {max([len(cluster) for cluster in clusters])}")
            print(f"Mean Cluster Size: {np.mean([len(cluster) for cluster in clusters])}")
        elif mode == 'pairs':
            self.train_sents = self._pairs(atrain, btrain, simtrain, threshold)
            self.dev_sents = self._pairs(adev, bdev, simdev, threshold)
        elif mode == 'triplets':
            self.train_sents = self._triplets(atrain, btrain, simtrain, threshold)
            self.dev_sents = self._triplets(adev, bdev, simdev, threshold)
        elif mode == 'classic':
            self.train_sents = np.array(list(zip(zip(atrain, btrain), simtrain)))
            self.dev_sents = np.array(list(zip(zip(adev, bdev), simdev)))
        else:
            raise ValueError("Mode can only be 'classic', 'clusters', 'pairs' or 'triplets'")

    def training_partition(self):
        np.random.shuffle(self.train_sents)
        # TODO add other modes
        if self.mode == 'classic':
            return SemEvalPairwisePartition(self.train_sents, self.batch_size, classes=True)
        else:
            return SemEvalClusterizedPartition(self.train_sents, self.batch_size)

    def test_partition(self):
        np.random.shuffle(self.dev_sents)
        # TODO add other modes
        if self.mode == 'classic':
            return SemEvalPairwisePartition(self.dev_sents, self.batch_size, classes=False)
        else:
            return SemEvalClusterizedPartition(self.dev_sents, self.batch_size)

    def _load_partition(self, partition):
        with open(join(self.path, partition, 'a.toks')) as afile, \
                open(join(self.path, partition, 'b.toks')) as bfile, \
                open(join(self.path, partition, 'sim.txt')) as simfile:
            a = [line.strip() for line in afile.readlines()]
            b = [line.strip() for line in bfile.readlines()]
            sim = [float(line.strip()) for line in simfile.readlines()]
            for i in range(len(a)):
                a[i], b[i] = self.pad_sent(a[i], b[i])
            if partition == 'train':
                sim = self.scores_to_probs(sim)
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


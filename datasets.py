#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from os.path import join
from tqdm import tqdm
from collections import deque


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


class MNISTPartition(SimDatasetPartition):
    
    def __init__(self, loader):
        super(MNISTPartition, self).__init__()
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
        testset = datasets.MNIST(path, download=True, train=False, transform=transform)
        self.train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)

    def training_partition(self):
        return MNISTPartition(self.train_loader)

    def test_partition(self):
        return MNISTPartition(self.test_loader)


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


class SemEvalSegment:

    @staticmethod
    def find_cluster(clusters, sentence):
        for i, cluster in enumerate(clusters):
            if sentence in cluster:
                return i
        return None

    def __init__(self, sents):
        self.sents = sents

    def clusters(self, other_segment, scores, threshold=2.5):
        """
        Consider scores as edge weights in a graph of sentences.
        Search for positive and negative pairs Breadth First Search
        """
        clusters = []
        for i, s in tqdm(enumerate(set(self.sents))):
            if SemEvalSegment.find_cluster(clusters, s) is not None:
                continue
            c = [s]
            added = {(i, self)}
            stack = deque()
            for j, x in enumerate(self.sents):
                if j != i and x == s:
                    stack.append((j, other_segment, self, True))
                    added.add((j, other_segment))
            while stack:  # is not empty
                # Retrieve next sentence from the stack
                j, seg, other_seg, equals_last = stack.popleft()
                other_sent = seg.sents[j]
                # Create the pair
                equals_this = False
                if scores[j] >= threshold:
                    if equals_last:
                        # A = B = C --> A = C (We're putting these 2 in the same cluster)
                        c.append(other_sent)
                        equals_this = True
                # Add dependencies
                for k, x in enumerate(seg.sents):
                    if k != j and (k, other_seg) not in added and x == other_sent:
                        stack.append((k, other_seg, seg, equals_this))
                        added.add((k, other_seg))
            if len(c) > 1:
                clusters.append(c)
        return clusters

    def pos_neg_pairs(self, other_segment, scores, threshold=2.5):
        """
        Consider scores as edge weights in a graph of sentences.
        Search for positive and negative pairs Breadth First Search
        """
        if isinstance(threshold, tuple):
            tlow, thigh = threshold
        else:
            tlow = threshold
            thigh = threshold
        pos, neg = [], []
        for i, s in tqdm(enumerate(set(self.sents))):
            added = {(i, self)}
            stack = deque()
            for j, x in enumerate(self.sents):
                if j != i and x == s:
                    stack.append((j, other_segment, self, True))
                    added.add((j, other_segment))
            while stack:  # is not empty
                # Retrieve next sentence from the stack
                j, seg, other_seg, equals_last = stack.popleft()
                other_sent = seg.sents[j]
                # Create the pair
                equals_this = False
                if scores[j] >= thigh:
                    if equals_last:
                        # A = B = C --> A = C
                        pos.append((s, other_sent))
                        equals_this = True
                    else:
                        # A != B and B = C --> A != C
                        neg.append((s, other_sent))
                elif scores[j] <= tlow and equals_last:
                    # A = B and B != C --> A != C
                    neg.append((s, other_sent))
                # Add dependencies
                for k, x in enumerate(seg.sents):
                    if k != j and (k, other_seg) not in added and x == other_sent:
                        stack.append((k, other_seg, seg, equals_this))
                        added.add((k, other_seg))
        return pos, neg


class SemEvalPartition(SimDatasetPartition):

    def __init__(self, data):
        super(SemEvalPartition, self).__init__()
        self.data = data
        self.iterator = iter(data)

    def nbatches(self):
        return len(self.data)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data)
            return next(self.iterator)


class SemEval(SimDataset):

    @staticmethod
    def unique_pairs(xs, ys, scores):
        seen = set()
        xunique, yunique, sunique = [], [], []
        for x, y, score in zip(xs, ys, scores):
            if (x, y) not in seen:
                seen.add((x, y))
                seen.add((y, x))
                xunique.append(x)
                yunique.append(y)
                sunique.append(score)
        return xunique, yunique, sunique

    @staticmethod
    def anchor_related_sents(anchor, pairs):
        anchor_pairs = [(x, y) for x, y in pairs if x == anchor or y == anchor]
        related = []
        for pos1, pos2 in anchor_pairs:
            if pos1 != anchor:
                related.append(pos1)
            else:
                related.append(pos2)
        return related

    @staticmethod
    def triplets(unique_sents, pos_pairs, neg_pairs):
        anchors, positives, negatives = [], [], []
        for anchor in tqdm(unique_sents):
            for positive in SemEval.anchor_related_sents(anchor, pos_pairs):
                for negative in SemEval.anchor_related_sents(anchor, neg_pairs):
                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)
        return anchors, positives, negatives

    @staticmethod
    def pairs(segment_a, segment_b, scores, threshold=(2, 3)):
        pos, neg = segment_a.pos_neg_pairs(segment_b, scores, threshold=threshold)
        return set(pos), set(neg)

    @staticmethod
    def load_partition(partition_path, mode, threshold):
        with open(join(partition_path, 'a.toks'), 'r') as file_a,\
                open(join(partition_path, 'b.toks'), 'r') as file_b,\
                open(join(partition_path, 'sim.txt'), 'r') as score_file:
            sents_a = [line.strip() for line in file_a.readlines()]
            sents_b = [line.strip() for line in file_b.readlines()]
            scores = [float(line.strip()) for line in score_file.readlines()]
            sents_a, sents_b, scores = SemEval.unique_pairs(sents_a, sents_b, scores)
            unique_sents = set(sents_a + sents_b)
            segment_a = SemEvalSegment(sents_a)
            segment_b = SemEvalSegment(sents_b)
            if mode == 'pairs':
                pos, neg = SemEval.pairs(segment_a, segment_b, scores, threshold)
                data = np.array([(s1, s2, 0) for s1, s2 in pos] + [(s1, s2, 1) for s1, s2 in neg])
            elif mode == 'triplets':
                pos, neg = SemEval.pairs(segment_a, segment_b, scores, threshold)
                anchors, positives, negatives = SemEval.triplets(unique_sents, pos, neg)
                data = np.array(list(zip(anchors, positives, negatives)))
            elif mode == 'auto':
                clusters = segment_a.clusters(segment_b, scores, threshold)
                data, lens = [], []
                for i, cluster in enumerate(clusters):
                    lens.append(len(cluster))
                    data += [(x, i) for x in cluster]
            else:
                raise ValueError("Mode can only be 'auto', 'pairs' or 'triplets'")
            np.random.shuffle(data)
        return data

    def __init__(self, path, batch_size, mode='auto', threshold=2.5):
        self.batch_size = batch_size
        self.train_sents = self.load_partition(join(path, 'train'), mode, threshold)
        # TODO uncomment this when dev partition is available
        # self.dev_sents = self.load_partition(join(path, 'dev'), mode, threshold)

    def training_partition(self):
        return SemEvalPartition(self.train_sents)

    def test_partition(self):
        # TODO use dev set
        return SemEvalPartition(self.train_sents)



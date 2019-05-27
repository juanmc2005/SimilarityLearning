#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.database import get_protocol
from pyannote.database import FileFinder


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
        # TODO dev
        self.test_gen = SpeechSegmentGenerator(extractor, protocol, subset='test', per_label=1,
                                               per_fold=batch_size, duration=0.2, parallel=2)

    def training_partition(self):
        return VoxCelebPartition(self.train_gen)

    def test_partition(self):
        return VoxCelebPartition(self.test_gen)

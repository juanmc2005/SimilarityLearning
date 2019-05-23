#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.database import get_protocol
from pyannote.database import FileFinder


def mnist(path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
    testset = datasets.MNIST(path, download=True, train=False, transform=transform)
    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


def voxceleb1():
    extractor = RawAudio(sample_rate=16000)
    preprocessors = {'audio': FileFinder()}
    protocol = get_protocol('VoxCeleb.SpeakerVerification.VoxCeleb1', preprocessors=preprocessors)
    train_gen = SpeechSegmentGenerator(extractor, protocol, subset='train', duration=1)
    dev_gen = SpeechSegmentGenerator(extractor, protocol, subset='test', duration=1)
    return train_gen, dev_gen


if __name__ == '__main__':
    train, test = voxceleb1()
    trloader = train()
    for _ in range(10):
        print(next(trloader))

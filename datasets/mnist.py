#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets.base import SimDataset, LoaderWrapperPartition, SimDatasetPartition


class MNIST(SimDataset):

    def __init__(self, path, batch_size):
        super(MNIST, self).__init__()
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        self.trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
        self.testset = datasets.MNIST(path, download=True, train=False, transform=transform)

    def training_partition(self) -> SimDatasetPartition:
        return LoaderWrapperPartition(DataLoader(self.trainset, self.batch_size, shuffle=True, num_workers=4))

    def dev_partition(self) -> SimDatasetPartition:
        return LoaderWrapperPartition(DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=4))

    def test_partition(self) -> SimDatasetPartition:
        return self.dev_partition()

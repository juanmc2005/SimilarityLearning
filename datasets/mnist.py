#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets.base import SimDataset, LoaderWrapperPartition, SimDatasetPartition
import random


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Borrowed from: https://github.com/galatolofederico/pytorch-balanced-batch
    """

    def __init__(self, dataset, labels=None):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if dataset_type is datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif dataset_type is datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


class MNIST(SimDataset):

    def __init__(self, path: str, batch_size: int, balance: bool):
        super(MNIST, self).__init__()
        self.batch_size = batch_size
        self.balance = balance
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        self.trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
        self.testset = datasets.MNIST(path, download=True, train=False, transform=transform)

    def training_partition(self) -> SimDatasetPartition:
        return LoaderWrapperPartition(DataLoader(self.trainset, self.batch_size,
                                                 sampler=BalancedBatchSampler(self.trainset) if self.balance else None,
                                                 shuffle=not self.balance, num_workers=4))

    def dev_partition(self) -> SimDatasetPartition:
        return LoaderWrapperPartition(DataLoader(self.testset, self.batch_size, shuffle=False, num_workers=4))

    def test_partition(self) -> SimDatasetPartition:
        return self.dev_partition()

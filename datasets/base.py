#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class SimDataset:

    def training_partition(self):
        raise NotImplementedError

    def dev_partition(self):
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

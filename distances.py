#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import spearmanr


# TODO remove this function and use pyannote function directly
def to_condensed(n, i, j):
    """
    Borrowed from pyannote: https://github.com/pyannote/pyannote-core
    Compute index in condensed pdist matrix
                V
        0 | . 0 1 2 3
     -> 1 | . . 4 5 6 <-   ==>   0 1 2 3 4 5 6 7 8 9
        2 | . . . 7 8                    ^
        3 | . . . . 9
        4 | . . . . .
           ----------
            0 1 2 3 4
    Parameters
    ----------
    n : int
        Number of inputs in squared pdist matrix
    i, j : `int` or `numpy.ndarray`
        Indices in squared pdist matrix
    Returns
    -------
    k : `int` or `numpy.ndarray`
        Index in condensed pdist matrix
    """
    i, j = np.array(i), np.array(j)
    if np.any(i == j):
        raise ValueError('i and j should be different.')
    i, j = np.minimum(i, j), np.maximum(i, j)
    return np.int64(i * n - i * i / 2 - 3 * i / 2 + j - 1)


class Distance:
    """
    A distance function implementing pairwise distance
    """
    
    def pdist(self, x):
        """
        Calculate the pairwise distance for a given batch
        :param x: a tensor of shape (N, d), where
            N = batch size
            d = feature dimension
        :return: a 1D tensor corresponding to the condensed triangular distance matrix
        """
        raise NotImplementedError("a Distance should implement 'pdist'")
    
    def sqdist_sum(self, x, y):
        """
        Calculate the squared distance between 2 batches and then return the sum
        :param x: a tensor of shape (N, d), where
            N = batch size
            d = feature dimension
        :param y: a tensor of shape (N, d), where
            N = batch size
            d = feature dimension
        :return: a tensor with a single value, the sum of the squared distances
        """
        raise NotImplementedError("a Distance should implement 'sqdist_sum'")
    
    def to_sklearn_metric(self):
        """
        Get the scikit-learn name for this function. This should return a string that
            can be used with sklearn functions
        :return: a string representing the metric that this Distance object implements
        """
        raise NotImplementedError("a Distance should implement 'to_sklearn_metric'")


class CosineDistance(Distance):
    """
    Cosine distance module using PyTorch's cosine_similarity to calculate pdist
    """
    
    def __init__(self):
        super(CosineDistance, self).__init__()
    
    def __str__(self):
        return 'Cosine Distance'
    
    def to_sklearn_metric(self):
        return 'cosine'
    
    def sqdist_sum(self, x, y):
        d = 1 - F.cosine_similarity(x, y, dim=1, eps=1e-8)
        return d.pow(2).sum()
    
    def pdist(self, x):
        nbatch, _ = x.size()
        distances = []
        for i in range(nbatch-1):
            d = 1. - F.cosine_similarity(
                x[i, :].expand(nbatch - 1 - i, -1),
                x[i+1:, :], dim=1, eps=1e-8)
            distances.append(d)
        return torch.cat(distances)


class EuclideanDistance(Distance):
    """
    Euclidean distance module using PyTorch's pdist,
        which already supports this distance
    """
    
    def __init__(self):
        super(EuclideanDistance, self).__init__()
    
    def __str__(self):
        return 'Euclidean Distance'
    
    def to_sklearn_metric(self):
        return 'euclidean'
    
    def sqdist_sum(self, x, y):
        return (x - y).pow(2).sum()
    
    def pdist(self, x):
        return F.pdist(x)


class KNNAccuracyMetric:
    """ TODO update docs
    Abstracts the accuracy calculation strategy. It uses a K Nearest Neighbors
        classifier fit with the embeddings produced for the training set,
        to determine to which class a given test embedding is assigned to.
    :param train_embeddings: a tensor of shape (N, d), where
            N = training set size
            d = embedding dimension
    :param train_y: a non one-hot encoded tensor of labels for the train embeddings
    :param distance: a Distance object for the KNN classifier
    """
    
    def __init__(self, distance):
        self.knn = KNeighborsClassifier(n_neighbors=1, metric=distance.to_sklearn_metric())
        self.correct, self.total = 0, 0

    def fit(self, embeddings, y):
        self.knn.fit(embeddings, y)
    
    def calculate_batch(self, embeddings, logits, y):
        predicted = self.knn.predict(embeddings)
        self.correct += (predicted == y).sum()
        self.total += y.shape[0]

    def get(self):
        metric = self.correct / self.total
        self.correct, self.total = 0, 0
        return metric


class SpearmanMetric:

    def __init__(self):
        self.predictions, self.targets = [], []

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        output = np.exp(logits)
        predicted = []
        for i in range(output.shape[0]):
            predicted.append(0 * output[i, 0] +
                             1 * output[i, 1] +
                             2 * output[i, 2] +
                             3 * output[i, 3] +
                             4 * output[i, 4] +
                             5 * output[i, 5])
        self.predictions.extend(predicted)
        self.targets.extend(list(y))

    def get(self):
        metric = spearmanr(self.predictions, self.targets)[0]
        self.predictions, self.targets = [], []
        return metric


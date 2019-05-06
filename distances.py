#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np


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
    
    def pdist(self, x):
        raise NotImplementedError("a Distance should implement 'pdist'")


class CosineDistance(Distance):
    
    def __init__(self):
        super(CosineDistance, self).__init__()
    
    def __str__(self):
        return 'Cosine Distance'
    
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
    
    def __init__(self):
        super(EuclideanDistance, self).__init__()
    
    def __str__(self):
        return 'Euclidean Distance'
    
    def pdist(self, x):
        return F.pdist(x)

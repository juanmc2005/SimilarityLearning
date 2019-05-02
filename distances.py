#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


class Distance:
    
    def pdist(self, x):
        raise NotImplementedError("a Distance should implement the method 'pdist'")


class CosineDistance(Distance):
    
    def __init__(self):
        super(CosineDistance, self).__init__()
    
    def __str__(self):
        return 'cosine distance'
    
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
        return 'euclidean distance'
    
    def pdist(self, x):
        return F.pdist(x)

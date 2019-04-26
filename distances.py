#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


class CosineDistance:
    
    def __call__(self, x1, x2):
        return 1 - F.cosine_similarity(x1, x2)
    
    def __str__(self):
        return 'cosine distance'


class EuclideanDistance:
    
    def __call__(self, x1, x2):
        return torch.norm(x1 - x2, dim=1)
    
    def __str__(self):
        return 'euclidean distance'

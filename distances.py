#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def cos_dist(x1, x2):
    return 1 - F.cosine_similarity(x1, x2)


def eucl_dist(X1, X2):
    return torch.sqrt(torch.sum(torch.pow(X1-X2, 2), 1))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from distances import eucl_dist, cos_dist


class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin=1.0, distance='euclidean'):
        super(ContrastiveLoss, self).__init__()
        print("Will optimize Contrastive Loss with {} distance".format(distance))
        self.margin = margin
        if distance == 'euclidean':
            self.dist_fn = eucl_dist
        elif distance == 'cosine':
            self.dist_fn = cos_dist
        else:
            raise ValueError('distance should be either cosine or euclidean')
    
    def forward(self, Y, X1, X2):
        dist = self.dist_fn(X1, X2)
        loss = (1-Y) * torch.pow(dist, 2) + Y * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return torch.sum(loss) / 2.0 / X1.size(0)
    
    def distance(self, x1, x2):
        return self.dist_fn(x1, x2)

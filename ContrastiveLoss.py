#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from distances import eucl_dist, cos_dist


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss module
    Reference: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    
    :param margin: the margin to separate feature vectors considered different
    :param distance: the base distance to use (either 'euclidean' or 'cosine')
    """
    
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
        """
        Calculate the contrastive loss measure
        
        :param Y: a non one-hot label tensor
        :param X1: a tensor
        :param X2: a tensor
        
        :return: the contrastive loss
        """
        # Calculate the distance between our 2 examples
        dist = self.dist_fn(X1, X2)
        # Calculate the loss based on the distance, using the margin
        loss = (1-Y) * torch.pow(dist, 2) + Y * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        # Return the sum of our losses normalized by the size of the batch
        return torch.sum(loss) / 2.0 / X1.size(0)
    
    def distance(self, x1, x2):
        """
        Apply the base distance function to 2 vectors (or batches of vectors)
        
        :param x1: a tensor of dimension (N, C)
        :param x2: a tensor of dimension (N, C)
        
        :return: the distance measure
        """
        return self.dist_fn(x1, x2)

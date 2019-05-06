#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss module
    Reference: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param device: a device where to execute the calculations
    :param margin: the margin to separate feature vectors considered different
    """
    
    def __init__(self, device, margin, distance):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.distance = distance
    
    def forward(self, x, y):
        """
        Calculate the contrastive loss
        :param x: a tensor corresponding to a batch of size (N, d), where
                  N = batch size, d = dimension of the feature vectors
        :param y: a non one-hot label tensor corresponding to the batch x
        :return: the contrastive loss
        """
        # First calculate the (euclidean) distances between every sample in the batch
        nbatch = x.size(0)
        dist = self.distance.pdist(x).to(self.device)
        # Calculate the ground truth Y corresponding to the pairs
        gt = []
        for i in range(nbatch-1):
            for j in range(i+1, nbatch):
                gt.append(int(y[i] != y[j]))
        gt = torch.Tensor(gt).float().to(self.device)
        # Calculate the losses as described in the paper
        loss = (1-gt) * torch.pow(dist, 2) + gt * torch.pow(torch.clamp(self.margin - dist, min=1e-8), 2)
        # Normalize by batch size
        return torch.sum(loss) / 2 / dist.size(0)
    
    def eval(self, x, y):
        nbatch = x.size(0)
        dist = self.distance.pdist(x).to(self.device)
        n = dist.size(0)
        gt = []
        for i in range(nbatch-1):
            for j in range(i+1, nbatch):
                gt.append(int(y[i] != y[j]))
        gt = torch.Tensor(gt).float().to(self.device)
        correct = np.sum([1 for i in range(n) if (gt[i] == 0 and dist[i] < self.margin) or (gt[i] == 1 and dist[i] >= self.margin)])
        return correct, n

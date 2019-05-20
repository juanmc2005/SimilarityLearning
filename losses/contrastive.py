#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from losses.base import BaseTrainer, Optimizer, Evaluator
from models import MNISTNet
from distances import EuclideanDistance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss module
    Reference: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param device: a device where to execute the calculations
    :param margin: the margin to separate feature vectors considered different
    :param distance: a Distance object to calculate our Dw
    """
    
    def __init__(self, device, margin, distance):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.distance = distance
    
    def forward(self, feat, logits, y):
        """
        Calculate the contrastive loss
        :param feat: a tensor corresponding to a batch of size (N, d), where
            N = batch size
            d = dimension of the feature vectors
        :param logits: unused, it's been kept for compatibility purposes
        :param y: a non one-hot label tensor corresponding to the batch
        :return: the contrastive loss
        """
        # First calculate the (euclidean) distances between every sample in the batch
        nbatch = feat.size(0)
        dist = self.distance.pdist(feat).to(self.device)
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


def contrastive_config(device, nfeat, margin=2, distance=EuclideanDistance()):
    model = MNISTNet(nfeat)
    optimizers = [optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 4, gamma=0.8)]
    return {
            'name': 'Contrastive Loss',
            'param_desc': f"m={margin} - {distance}",
            'model': model,
            'loss': ContrastiveLoss(device, margin, distance),
            'optim': Optimizer(optimizers, schedulers),
            'test_distance': distance
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxCenterLoss(nn.Module):
    """
    Cross Entropy + Center Loss module
    Reference: https://kpzhang93.github.io/papers/eccv2016.pdf
    :param device: a device in which to run the computation
    :param nfeat: the number of features in the embedding
    :param nclass: the number of classes
    :param loss_weight: a value for the lambda parameter described in the paper,
        to use as weight for the center loss
    :param distance: a distance object to use when calculating distances to the centers
    """
    
    def __init__(self, device, nfeat, nclass, loss_weight, distance):
        super(SoftmaxCenterLoss, self).__init__()
        self.loss_weight = loss_weight
        self.center = CenterLoss(nclass, nfeat, distance).to(device)
        self.nll = nn.NLLLoss().to(device)
    
    def forward(self, feat, logits, y):
        """
        Calculate the total center loss, with cross entropy supervision
        :param feat: a tensor corresponding to an embedding batch of size (N, d), where
            N = batch size
            d = dimension of the feature vectors
        :param logits: a tensor corresponding to a logits batch of size (N, c), where
            N = batch size
            c = number of classes
        :param y: a non one-hot label tensor corresponding to the batch
        :return: the loss value for this batch
        """
        return self.nll(logits, y) + self.loss_weight * self.center(feat, logits, y)
    
    def center_parameters(self):
        return self.center.parameters()


class CenterLoss(nn.Module):
    """
    Center Loss module
    Reference: https://kpzhang93.github.io/papers/eccv2016.pdf
    :param nfeat: the number of features in the embedding
    :param nclass: the number of classes
    :param distance: a distance object to use when calculating distances to the centers
    """
    
    def __init__(self, nclass, nfeat, distance):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(nclass, nfeat))
        self.nfeat = nfeat
        self.distance = distance
    
    def forward(self, feat, logits, y):
        """
        Calculate the center loss
        :param feat: a tensor corresponding to an embedding batch of size (N, d), where
            N = batch size
            d = dimension of the feature vectors
        :param logits: unused, it's been kept for compatibility purposes
        :param y: a non one-hot label tensor corresponding to the batch
        :return: the center loss value for this batch
        """
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # Select appropriate centers for this batch's labels
        centers_batch = self.centers.index_select(0, y.long())
        # Return the sum of the squared distance normalized by the batch size
        return self.distance.sqdist_sum(feat, centers_batch) / 2.0 / batch_size


class CenterLinear(nn.Module):
    """
    Center linear layer module
    Reference: https://kpzhang93.github.io/papers/eccv2016.pdf
    :param nfeat: the number of features in the embedding
    :param nclass: the number of classes
    """
    
    def __init__(self, nfeat, nclass):
        super(CenterLinear, self).__init__()
        # No bias to distribute centers in a circular manner (for euclidean distance)
        self.linear = nn.Linear(nfeat, nclass, bias=False)
    
    def forward(self, x, y):
        """
        Apply the linear transformation and softmax
        :param x: an embedding batch
        :param y: a non one-hot label batch
        :return: a tensor with class probabilities for this batch
        """
        return F.log_softmax(self.linear(x), dim=1)

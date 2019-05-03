#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import squareform


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


class ArcLinear(nn.Module):
    """
    Additive Angular Margin loss module (ArcFace)
    Reference: https://arxiv.org/pdf/1801.07698.pdf
    :param nfeat: the number of features in the embedding
    :param nclass: the number of classes
    :param margin: the margin to separate classes in angular space
    :param s: the scaling factor for the feature vector
    """
    
    def __init__(self, nfeat, nclass, margin, s):
        super(ArcLinear, self).__init__()
        eps = 1e-4
        self.min_cos = eps - 1
        self.max_cos = 1 - eps
        self.nclass = nclass
        self.margin = margin
        self.s = s
        self.W = nn.Parameter(torch.Tensor(nclass, nfeat))
        nn.init.xavier_uniform_(self.W)
        self.last_logits, self.last_y = None, None
    
    def forward(self, x, y):
        """
        Apply the angular margin transformation
        :param x: a feature vector batch
        :param y: a non one-hot label batch
        :return: the value for the Additive Angular Margin loss
        """
        # Normalize the feature vectors and W
        xnorm = F.normalize(x)
        Wnorm = F.normalize(self.W)
        y = y.long().view(-1, 1)
        # Calculate cosθj (the logits)
        cos_theta_j = torch.matmul(xnorm, torch.transpose(Wnorm, 0, 1))
        # Get the cosθ corresponding to the classes
        cos_theta_yi = cos_theta_j.gather(1, y)
        # For numerical stability
        cos_theta_yi = cos_theta_yi.clamp(min=self.min_cos, max=self.max_cos)
        # Get the angle separating xi and Wyi
        theta_yi = torch.acos(cos_theta_yi)
        # Apply the margin to the angle
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # One hot encode  y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, y, 1.0)
        # Project margin differences into cosθj
        cos_theta_j += one_hot * (cos_theta_yi_margin - cos_theta_yi)
        # Apply the scaling
        cos_theta_j = self.s * cos_theta_j
        self.last_logits, self.last_y = cos_theta_j, y
        return cos_theta_j
    
    def eval_last_forward(self):
        _, predicted = torch.max(self.last_logits.data, 1)
        return (predicted == self.last_y.data).sum(), self.last_y.size(0)
    

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
        self.last_pdist, self.last_gt = None, None
    
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
        self.last_pdist, self.last_gt = dist, gt
        # Calculate the losses as described in the paper
        loss = (1-gt) * torch.pow(dist, 2) + gt * torch.pow(torch.clamp(self.margin - dist, min=1e-8), 2)
        # Normalize by batch size
        return torch.sum(loss) / 2 / dist.size(0)
    
    def eval_last_forward(self):
        dist = self.last_pdist
        gt = self.last_gt
        correct = torch.sum((gt == 0 and dist < self.margin) or (gt == 1 and dist >= self.margin))
        return correct, self.last_pdist.size(0)
    
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


class TripletLoss(nn.Module):

    def __init__(self, device, margin, distance):
        super(TripletLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.distance = distance
        self.last_dpos, self.last_dneg = None, None
    
    def batch_triplets(self, y, distance):
        anchors, positives, negatives = [], [], []
        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):
                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue
                for negative, y_negative in enumerate(y):
                    if y_negative == y_anchor:
                        continue
                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)
        return anchors, positives, negatives
    
    def batch_negative_triplets(self, y, distances):
        anchors, positives, negatives = [], [], []
        distances = squareform(distances.detach().cpu().numpy())
        y = y.cpu().numpy()
        for anchor, y_anchor in enumerate(y):
            # hardest negative
            d = distances[anchor]
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])
            for positive in np.where(y == y_anchor)[0]:
                if positive == anchor:
                    continue
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
        return anchors, positives, negatives
    
    def batch_hardest_triplets(self, y, distances):
        anchors, positives, negatives = [], [], []
        distances = squareform(distances.detach().cpu().numpy())
        y = y.cpu().numpy()
        for anchor, y_anchor in enumerate(y):
            d = distances[anchor]
            # hardest positive
            pos = np.where(y == y_anchor)[0]
            pos = [p for p in pos if p != anchor]
            positive = int(pos[np.argmax(d[pos])])
            # hardest negative
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
        return anchors, positives, negatives
    
    def calculate_distances(self, x, y, sampling_strategy):
        n = x.size(0)
        dist = self.distance.pdist(x).to(self.device)
        anchors, positives, negatives = sampling_strategy(y, dist)
        pos = to_condensed(n, anchors, positives)
        neg = to_condensed(n, anchors, negatives)
        return dist[pos], dist[neg]
    
    def forward(self, x, y):
        dpos, dneg = self.calculate_distances(x, y, self.batch_triplets)
        loss = dpos - dneg + self.margin
        self.last_dpos = dpos
        self.last_dneg = dneg
        return torch.mean(torch.clamp(loss, min=1e-8))
    
    def eval_last_forward(self):
        correct_positives = torch.sum(self.last_dpos < self.margin)
        correct_negatives = torch.sum(self.last_dneg >= self.margin)
        return correct_positives + correct_negatives, len(self.last_dpos) + len(self.last_dneg)
    
    def eval(self, x, y):
        dpos, dneg = self.calculate_distances(x, y, self.batch_triplets)
        correct_positives = torch.sum(dpos < self.margin)
        correct_negatives = torch.sum(dneg >= self.margin)
        return correct_positives + correct_negatives, len(dpos) + len(dneg)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
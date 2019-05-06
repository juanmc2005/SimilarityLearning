#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import squareform
from distances import to_condensed
from CenterLoss import CenterLoss


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
        return cos_theta_j
    

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


class TripletLoss(nn.Module):

    def __init__(self, device, margin, distance):
        super(TripletLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.distance = distance
    
    def batch_triplets(self, y):
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
    
    def calculate_distances(self, x, y):
        n = x.size(0)
        dist = self.distance.pdist(x).to(self.device)
        anchors, positives, negatives = self.batch_triplets(y)
        pos = to_condensed(n, anchors, positives)
        neg = to_condensed(n, anchors, negatives)
        return dist[pos], dist[neg]
    
    def forward(self, x, y):
        dpos, dneg = self.calculate_distances(x, y)
        loss = dpos - dneg + self.margin
        return torch.mean(torch.clamp(loss, min=1e-8))
    
    def eval(self, x, y):
        dpos, dneg = self.calculate_distances(x, y)
        correct_positives = torch.sum(dpos < self.margin)
        correct_negatives = torch.sum(dneg >= self.margin)
        return correct_positives + correct_negatives, len(dpos) + len(dneg)
    

class SoftmaxCenterLoss(nn.Module):
    
    def __init__(self, device, nfeat, nclass, loss_weight):
        super(SoftmaxCenterLoss, self).__init__()
        self.loss_weight = loss_weight
        self.center = CenterLoss(nclass, nfeat).to(device)
        self.nll = nn.NLLLoss().to(device)
    
    def forward(self, data, y):
        feat, preds = data
        return self.nll(preds, y) + self.loss_weight * self.center(y, feat)
    
    def center_parameters(self):
        return self.center.parameters()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
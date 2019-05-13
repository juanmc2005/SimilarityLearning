#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.distance import squareform
from losses.base import BaseTrainer
from distances import to_condensed, EuclideanDistance
from models import MNISTNet


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
        anchors, positives, negatives = self.batch_negative_triplets(y)
        pos = to_condensed(n, anchors, positives)
        neg = to_condensed(n, anchors, negatives)
        return dist[pos], dist[neg]
    
    def forward(self, feat, logits, y):
        dpos, dneg = self.calculate_distances(feat, y)
        loss = dpos - dneg + self.margin
        return torch.mean(torch.clamp(loss, min=1e-8))


class TripletTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, margin=0.2, distance=EuclideanDistance(), batch_size=25):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)
        super(TripletTrainer, self).__init__(
                MNISTNet(nfeat),
                device,
                TripletLoss(device, margin, distance),
                distance,
                train_loader,
                test_loader)
        self.margin = margin
        self.distance = distance
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 3, gamma=0.8)
        ]
    
    def __str__(self):
        return 'Triplet Loss'
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def describe_params(self):
        return f"m={self.margin} - {self.distance}"
    
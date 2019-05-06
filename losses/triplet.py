#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import squareform
from distances import to_condensed


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
    
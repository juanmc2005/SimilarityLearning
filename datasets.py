#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch


class ContrastiveDataset:
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        n = Y.size(0)
        fst, snd = [], []
        Ypairs = []
        for i in range(n-1):
            added = []
            for j in range(i+1, n):
                label = Y[j]
                if i != j and label not in added:
                    fst.append(i)
                    snd.append(j)
                    Ypairs.append(0 if Y[i] == Y[j] else 1)
                    added.append(label)
                if len(added) == 10:
                    break
        self.pairs1 = torch.LongTensor(fst)
        self.pairs2 = torch.LongTensor(snd)
        self.contrastiveY = torch.FloatTensor(Ypairs)
        self.size = self.contrastiveY.size(0)
        
    def to(self, device):
        self.X.to(device)
        self.pairs1.to(device)
        self.pairs2.to(device)
        self.contrastiveY.to(device)
        return self
        
    def __getitem__(self, i):
        i1 = self.pairs1[i]
        i2 = self.pairs2[i]
        return (self.X[i1], self.X[i2], self.contrastiveY[i], self.Y[i1], self.Y[i2])
    
    def __len__(self):
        return self.size

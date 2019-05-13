#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from losses.base import BaseTrainer
from models import MNISTNet
from distances import EuclideanDistance

class SoftmaxCenterLoss(nn.Module):
    
    def __init__(self, device, nfeat, nclass, loss_weight, distance):
        super(SoftmaxCenterLoss, self).__init__()
        self.loss_weight = loss_weight
        self.center = CenterLoss(nclass, nfeat, distance).to(device)
        self.nll = nn.NLLLoss().to(device)
    
    def forward(self, feat, logits, y):
        return self.nll(logits, y) + self.loss_weight * self.center(y, feat)
    
    def center_parameters(self):
        return self.center.parameters()


class CenterLoss(nn.Module):
    
    def __init__(self, nclass, nfeat, distance):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(nclass, nfeat))
        self.nfeat = nfeat
        self.distance = distance
    
    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        centers_batch = self.centers.index_select(0, label.long())
        return self.distance.sqdist_sum(feat, centers_batch) / 2.0 / batch_size


class CenterLinear(nn.Module):
    
    def __init__(self, nfeat, nclass):
        super(CenterLinear, self).__init__()
        self.linear = nn.Linear(nfeat, nclass, bias=False)
    
    def forward(self, x, y):
        return F.log_softmax(self.linear(x), dim=1)


class CenterTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, nclass, loss_weight=1, distance=EuclideanDistance(), batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(CenterTrainer, self).__init__(
                MNISTNet(nfeat, loss_module=CenterLinear(nfeat, nclass)),
                device,
                SoftmaxCenterLoss(device, nfeat, nclass, loss_weight, distance),
                distance,
                train_loader,
                test_loader)
        self.loss_weight = loss_weight
        self.distance = distance
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005),
                optim.SGD(self.loss_fn.center_parameters(), lr=0.5)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 20, gamma=0.8)
        ]
    
    def __str__(self):
        return 'Center Loss'
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def describe_params(self):
        return f"λ={self.loss_weight} - {self.distance}"

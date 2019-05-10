#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MNISTNet(nn.Module):
    
    def __init__(self, nfeat, loss_module=None):
        super(MNISTNet, self).__init__()
        self.loss_module = loss_module
        self.out_dim = 128*3*3
        self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.PReLU(),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.PReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.PReLU(),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.PReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.PReLU(),
                nn.Conv2d(128, 128, kernel_size=5, padding=2),
                nn.PReLU(),
                nn.MaxPool2d(2),
                Flatten(),
                nn.Linear(self.out_dim, nfeat),
                nn.PReLU()
        )
    
    def forward(self, x, y):
        feat = self.net(x)
        logits = self.loss_module(feat, y) if self.loss_module is not None else None
        return feat, logits
    
    def net_params(self):
        return self.net.parameters()
    
    def loss_params(self):
        return self.loss_module.parameters()

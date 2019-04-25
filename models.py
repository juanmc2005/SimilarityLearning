#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F

class CommonNet(nn.Module):
    def __init__(self):
        super(CommonNet, self).__init__()
        self.out_dim = 128*3*3
        self.conv = nn.Sequential(
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
                nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.conv(x).view(-1, self.out_dim)

class CenterNet(nn.Module):
    def __init__(self):
        super(CenterNet, self).__init__()
        self.common = CommonNet()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(self.common.out_dim, 2)
        self.ip2 = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        x = self.common(x)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2, dim=1)

class ContrastiveNet(nn.Module):
    def __init__(self):
        super(ContrastiveNet, self).__init__()
        self.common = CommonNet()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(self.common.out_dim, 10)
        self.ip2 = nn.Linear(10, 2, bias=False)

    def forward(self, x):
        x = self.common(x)
        x = self.preluip1(self.ip1(x))
        return self.ip2(x)

class ArcNet(nn.Module):
    def __init__(self):
        super(ArcNet, self).__init__()
        self.common = CommonNet()
        self.bn1 = nn.BatchNorm1d(self.common.out_dim)
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(self.common.out_dim, 2)

    def forward(self, x):
        x = self.bn1(self.common(x))
        return self.preluip1(self.ip1(x))

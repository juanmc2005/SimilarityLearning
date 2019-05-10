#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
from losses.arcface import ArcLinear
from losses.coco import CocoLinear


class CommonNet(nn.Module):
    
    def __init__(self, nfeat):
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
        self.dense = nn.Linear(self.out_dim, nfeat)
        self.prelu = nn.PReLU()
    
    def forward(self, x, y):
        x = self.conv(x).view(-1, self.out_dim)
        return self.prelu(self.dense(x)), None


# FIXME these models can be unified by parameterizing the classification layer

class CenterNet(nn.Module):
    
    def __init__(self, nfeat, nclass):
        super(CenterNet, self).__init__()
        self.common = CommonNet(nfeat)
        self.linear = nn.Linear(nfeat, nclass, bias=False)

    def forward(self, x, y):
        feat, _ = self.common(x, y)
        logits = self.linear(feat)
        return feat, F.log_softmax(logits, dim=1)


class ArcNet(nn.Module):
    
    def __init__(self, nfeat, nclass, margin, s):
        super(ArcNet, self).__init__()
        self.common = CommonNet(nfeat)
        self.arc = ArcLinear(nfeat, nclass, margin, s)

    def forward(self, x, y):
        feat, _ = self.common(x, y)
        logits = self.arc(feat, y)
        return feat, logits
    
    def common_params(self):
        return self.common.parameters()
    
    def arc_params(self):
        return self.arc.parameters()


class CocoNet(nn.Module):
    
    def __init__(self, nfeat, nclass, alpha):
        super(CocoNet, self).__init__()
        self.common = CommonNet(nfeat)
        self.coco = CocoLinear(nfeat, nclass, alpha)

    def forward(self, x, y):
        feat, _ = self.common(x, y)
        logits = self.coco(feat, y)
        return feat, logits
    
    def common_params(self):
        return self.common.parameters()
    
    def coco_params(self):
        return self.coco.parameters()

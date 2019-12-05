# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class CocoLinear(nn.Module):
    
    def __init__(self, nfeat, nclass, alpha):
        super(CocoLinear, self).__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(nclass, nfeat))
    
    def forward(self, x, y):
        cnorm = F.normalize(self.centers)
        xnorm = self.alpha * F.normalize(x)
        logits = torch.matmul(xnorm, torch.transpose(cnorm, 0, 1))
        return logits

    def predict(self, x):
        return self.forward(x, None)

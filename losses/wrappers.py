# -*- coding: utf-8 -*-
import torch.nn as nn


class LossWrapper(nn.Module):
    
    def __init__(self, loss):
        super(LossWrapper, self).__init__()
        self.loss = loss
    
    def forward(self, feat, logits, y):
        return self.loss(logits, y)


class PWIMPairwiseClassification(nn.Module):

    def __init__(self, nfeat_sent):
        super(PWIMPairwiseClassification, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(2*nfeat_sent, 128), nn.Tanh(), nn.Linear(128, 6), nn.LogSoftmax())

    def forward(self, x, y):
        output = self.mlp(x)
        output = output.view(-1, 6)
        return output

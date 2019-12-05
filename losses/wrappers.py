# -*- coding: utf-8 -*-
import torch.nn as nn


class LossWrapper(nn.Module):
    
    def __init__(self, loss):
        super(LossWrapper, self).__init__()
        self.loss = loss
    
    def forward(self, feat, logits, y):
        return self.loss(logits, y)


class PredictLossModuleWrapper(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.mod = module

    def forward(self, x, y):
        return self.mod.predict(x)


class STSBaselineClassifier(nn.Module):

    def __init__(self, nfeat_sent):
        super(STSBaselineClassifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(4 * nfeat_sent, 128), nn.Tanh(), nn.Linear(128, 6), nn.LogSoftmax())

    def forward(self, x, y):
        output = self.mlp(x)
        output = output.view(-1, 6)
        return output


class SNLIClassifier(nn.Module):

    def __init__(self, nfeat_sent, nclass):
        super(SNLIClassifier, self).__init__()
        self.nclass = nclass
        self.mlp = nn.Sequential(nn.Linear(2 * nfeat_sent, 512), nn.Linear(512, nclass), nn.LogSoftmax())

    def forward(self, x, y):
        return self.mlp(x).view(-1, self.nclass)

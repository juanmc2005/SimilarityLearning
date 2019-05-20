# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from losses.base import Optimizer
from losses.center import CenterLinear
from models import MNISTNet
from distances import CosineDistance


class LossWrapper(nn.Module):
    
    def __init__(self, loss):
        super(LossWrapper, self).__init__()
        self.loss = loss
    
    def forward(self, feat, logits, y):
        return self.loss(logits, y)


def softmax_config(device, nfeat, nclass):
    model = MNISTNet(nfeat, loss_module=CenterLinear(nfeat, nclass))
    optimizers = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 10, gamma=0.5)]
    return {
            'name': 'Cross Entropy',
            'param_desc': None,
            'model': model,
            'loss': LossWrapper(nn.NLLLoss().to(device)),
            'optim': Optimizer(optimizers, schedulers),
            'test_distance': CosineDistance()
    }

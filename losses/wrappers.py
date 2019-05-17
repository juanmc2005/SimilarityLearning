# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from losses.base import BaseTrainer, TrainingConfig
from losses.center import CenterLinear
from models import MNISTNet
from distances import CosineDistance


class LossWrapper(nn.Module):
    
    def __init__(self, loss):
        super(LossWrapper, self).__init__()
        self.loss = loss
    
    def forward(self, feat, logits, y):
        return self.loss(logits, y)


def softmax_trainer(train_loader, test_loader, device, nfeat, nclass, callbacks):
    model = MNISTNet(nfeat, loss_module=CenterLinear(nfeat, nclass))
    optimizers = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)]
    config = TrainingConfig(
            name='Cross Entropy',
            optimizers=optimizers,
            schedulers=[lr_scheduler.StepLR(optimizers[0], 10, gamma=0.5)])
    return BaseTrainer(model, device, LossWrapper(nn.NLLLoss().to(device)), CosineDistance(),
            train_loader, test_loader, config, callbacks)

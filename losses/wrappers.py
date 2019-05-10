# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from losses.base import BaseTrainer
from losses.center import CenterLinear
from models import MNISTNet
from distances import CosineDistance


class LossWrapper(nn.Module):
    
    def __init__(self, loss):
        super(LossWrapper, self).__init__()
        self.loss = loss
    
    def forward(self, feat, logits, y):
        return self.loss(logits, y)


class SoftmaxTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, nclass, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(SoftmaxTrainer, self).__init__(
                MNISTNet(nfeat, loss_module=CenterLinear(nfeat, nclass)),
                device,
                LossWrapper(nn.NLLLoss().to(device)),
                CosineDistance(),
                train_loader,
                test_loader)
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 10, gamma=0.5)
        ]
    
    def __str__(self):
        return 'Cross Entropy'
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers

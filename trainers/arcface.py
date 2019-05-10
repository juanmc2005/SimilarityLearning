# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from trainers.base import BaseTrainer
from models import ArcNet
from losses.wrappers import LossWrapper
from distances import CosineDistance

class ArcTrainer(BaseTrainer):

    def __init__(self, trainset, testset, device, nfeat, nclass, margin=0.2, s=7.0, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(ArcTrainer, self).__init__(
                ArcNet(nfeat, nclass, margin, s),
                device,
                LossWrapper(nn.CrossEntropyLoss().to(device)),
                CosineDistance(),
                train_loader,
                test_loader)
        self.margin = margin
        self.s = s
        self.optimizers = [
                optim.SGD(self.model.common_params(), lr=0.005, momentum=0.9, weight_decay=0.0005),
                optim.SGD(self.model.arc_params(), lr=0.01)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 8, gamma=0.2),
                lr_scheduler.StepLR(self.optimizers[1], 8, gamma=0.2)
        ]
    
    def __str__(self):
        return 'ArcFace Loss'
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def describe_params(self):
        return f"m={self.margin} s={self.s}"

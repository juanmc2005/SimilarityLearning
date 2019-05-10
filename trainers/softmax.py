# -*- coding: utf-8 -*-
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from trainers.base import BaseTrainer
from losses.wrappers import LossWrapper
from models import CenterNet
from distances import CosineDistance


class SoftmaxTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, nclass, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(SoftmaxTrainer, self).__init__(
                CenterNet(nfeat, nclass),
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

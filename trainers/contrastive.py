# -*- coding: utf-8 -*-
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from trainers.base import BaseTrainer
from models import CommonNet
from losses.contrastive import ContrastiveLoss
from distances import EuclideanDistance


class ContrastiveTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, margin=2.0, distance=EuclideanDistance(), batch_size=80):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)
        super(ContrastiveTrainer, self).__init__(
                CommonNet(nfeat),
                device,
                ContrastiveLoss(device, margin, distance),
                distance,
                train_loader,
                test_loader)
        self.margin = margin
        self.distance = distance
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 2, gamma=0.8)
        ]
    
    def __str__(self):
        return 'Contrastive Loss'
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def describe_params(self):
        return f"m={self.margin} - {self.distance}"

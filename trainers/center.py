# -*- coding: utf-8 -*-
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from trainers.base import BaseTrainer
from losses.center import SoftmaxCenterLoss
from models import CenterNet
from distances import EuclideanDistance


class CenterTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, nclass, loss_weight=1, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(CenterTrainer, self).__init__(
                CenterNet(nfeat, nclass),
                device,
                SoftmaxCenterLoss(device, nfeat, nclass, loss_weight),
                EuclideanDistance(),
                train_loader,
                test_loader)
        self.loss_weight = loss_weight
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005),
                optim.SGD(self.loss_fn.center_parameters(), lr=0.5)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 20, gamma=0.8)
        ]
    
    def __str__(self):
        return 'Center Loss'
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def describe_params(self):
        return f"λ={self.loss_weight}"

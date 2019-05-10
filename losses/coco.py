# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from models import MNISTNet
from losses.base import BaseTrainer
from losses.wrappers import LossWrapper
from distances import CosineDistance

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


class CocoTrainer(BaseTrainer):

    def __init__(self, trainset, testset, device, nfeat, nclass, alpha=6.25, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(CocoTrainer, self).__init__(
                MNISTNet(nfeat, loss_module=CocoLinear(nfeat, nclass, alpha)),
                device,
                LossWrapper(nn.CrossEntropyLoss().to(device)),
                CosineDistance(),
                train_loader,
                test_loader)
        self.alpha = alpha
        self.optimizers = [
                optim.SGD(self.model.net_params(), lr=0.001, momentum=0.9, weight_decay=0.0005),
                optim.SGD(self.model.loss_params(), lr=0.01, momentum=0.9)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 10, gamma=0.5)
        ]
    
    def __str__(self):
        return 'CoCo Loss'
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def describe_params(self):
        return f"α={self.alpha}"

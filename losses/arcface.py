#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from losses.base import BaseTrainer
from models import ArcNet
from losses.wrappers import LossWrapper
from distances import CosineDistance


class ArcLinear(nn.Module):
    """
    Additive Angular Margin loss module (ArcFace)
    Reference: https://arxiv.org/pdf/1801.07698.pdf
    :param nfeat: the number of features in the embedding
    :param nclass: the number of classes
    :param margin: the margin to separate classes in angular space
    :param s: the scaling factor for the feature vector
    """
    
    def __init__(self, nfeat, nclass, margin, s):
        super(ArcLinear, self).__init__()
        eps = 1e-4
        self.min_cos = eps - 1
        self.max_cos = 1 - eps
        self.nclass = nclass
        self.margin = margin
        self.s = s
        self.W = nn.Parameter(torch.Tensor(nclass, nfeat))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, x, y):
        """
        Apply the angular margin transformation
        :param x: a feature vector batch
        :param y: a non one-hot label batch
        :return: the value for the Additive Angular Margin loss
        """
        # Normalize the feature vectors and W
        xnorm = F.normalize(x)
        Wnorm = F.normalize(self.W)
        y = y.long().view(-1, 1)
        # Calculate cosθj (the logits)
        cos_theta_j = torch.matmul(xnorm, torch.transpose(Wnorm, 0, 1))
        # Get the cosθ corresponding to the classes
        cos_theta_yi = cos_theta_j.gather(1, y)
        # For numerical stability
        cos_theta_yi = cos_theta_yi.clamp(min=self.min_cos, max=self.max_cos)
        # Get the angle separating xi and Wyi
        theta_yi = torch.acos(cos_theta_yi)
        # Apply the margin to the angle
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # One hot encode  y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, y, 1.0)
        # Project margin differences into cosθj
        cos_theta_j += one_hot * (cos_theta_yi_margin - cos_theta_yi)
        # Apply the scaling
        cos_theta_j = self.s * cos_theta_j
        return cos_theta_j


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

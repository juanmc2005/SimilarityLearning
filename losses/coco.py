# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from models import MNISTNet
from losses.base import BaseTrainer, TrainingConfig
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


def coco_trainer(train_loader, test_loader, device, nfeat, nclass, callbacks, alpha=6.25):
    model = MNISTNet(nfeat, loss_module=CocoLinear(nfeat, nclass, alpha))
    loss_fn = LossWrapper(nn.CrossEntropyLoss().to(device))
    optimizers = [
            optim.SGD(model.net_params(), lr=0.001, momentum=0.9, weight_decay=0.0005),
            optim.SGD(model.loss_params(), lr=0.01, momentum=0.9)]
    config = TrainingConfig(
            name='CoCo Loss',
            optimizers=optimizers,
            schedulers=[lr_scheduler.StepLR(optimizers[0], 10, gamma=0.5)],
            param_description=f"α={alpha}")
    return BaseTrainer(model, device, loss_fn, CosineDistance(),
                       train_loader, test_loader, config, callbacks)

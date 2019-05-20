# -*- coding: utf-8 -*-
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import MNISTNet
from distances import CosineDistance, EuclideanDistance
from losses.base import Optimizer
from losses.center import CenterLinear, SoftmaxCenterLoss
from losses.wrappers import LossWrapper
from losses.arcface import ArcLinear
from losses.coco import CocoLinear
from losses.contrastive import ContrastiveLoss
from losses.triplet import TripletLoss, BatchAll


def config(name, param_desc, model, loss, optims, scheds, test_dist):
    return {
            'name': name,
            'param_desc': param_desc,
            'model': model,
            'loss': loss,
            'optim': Optimizer(optims, scheds),
            'test_distance': test_dist
    }


def softmax(device, nfeat, nclass):
    model = MNISTNet(nfeat, loss_module=CenterLinear(nfeat, nclass))
    loss = LossWrapper(nn.NLLLoss().to(device))
    optimizers = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 10, gamma=0.5)]
    return config('Cross Entropy', None, model, loss, optimizers, schedulers, CosineDistance())


def arcface(device, nfeat, nclass, margin=0.2, s=7.0):
    model = MNISTNet(nfeat, loss_module=ArcLinear(nfeat, nclass, margin, s))
    loss = LossWrapper(nn.CrossEntropyLoss().to(device))
    optimizers = [optim.SGD(model.net_params(), lr=0.005, momentum=0.9, weight_decay=0.0005),
                  optim.SGD(model.loss_params(), lr=0.01)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 8, gamma=0.6),
                  lr_scheduler.StepLR(optimizers[1], 8, gamma=0.8)]
    return config('ArcFace Loss', f"m={margin} s={s}", model, loss, optimizers, schedulers, CosineDistance())


def center(device, nfeat, nclass, lweight=1, distance=EuclideanDistance()):
    model = MNISTNet(nfeat, loss_module=CenterLinear(nfeat, nclass))
    loss = SoftmaxCenterLoss(device, nfeat, nclass, lweight, distance)
    optimizers = [optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005),
                  optim.SGD(loss.center_parameters(), lr=0.5)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 20, gamma=0.8)]
    return config('Center Loss', f"λ={lweight} - {distance}", model, loss, optimizers, schedulers, distance)


def coco(device, nfeat, nclass, alpha=6.25):
    model = MNISTNet(nfeat, loss_module=CocoLinear(nfeat, nclass, alpha))
    loss = LossWrapper(nn.CrossEntropyLoss().to(device))
    optimizers = [optim.SGD(model.net_params(), lr=0.001, momentum=0.9, weight_decay=0.0005),
                  optim.SGD(model.loss_params(), lr=0.01, momentum=0.9)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 10, gamma=0.5)]
    return config('CoCo Loss', f"α={alpha}", model, loss, optimizers, schedulers, CosineDistance())


def contrastive(device, nfeat, margin=2, distance=EuclideanDistance()):
    model = MNISTNet(nfeat)
    loss = ContrastiveLoss(device, margin, distance)
    optimizers = [optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 4, gamma=0.8)]
    return config('Contrastive Loss', f"m={margin} - {distance}", model, loss, optimizers, schedulers, distance)


def triplet(device, nfeat, margin=2, distance=EuclideanDistance(), sampling=BatchAll()):
    model = MNISTNet(nfeat)
    loss = TripletLoss(device, margin, distance, sampling)
    optimizers = [optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)]
    schedulers = [lr_scheduler.StepLR(optimizers[0], 5, gamma=0.8)]
    return config('Triplet Loss', f"m={margin} - {distance}", model, loss, optimizers, schedulers, distance)

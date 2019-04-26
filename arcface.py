#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcLinear(nn.Module):
    """
    Additive Angular Margin loss module (ArcFace)
    Implementation borrowed from: https://github.com/egcode/pytorch-losses
    Reference: https://arxiv.org/pdf/1801.07698.pdf
    
    :param num_classes: the number of classes
    :param feat_dim: the number of features in the embedding
    :param device: the device in which the module will run
    :param s: the scaling factor for the feature vector
    :param m: the margin to separate classes in angular space
    """
    
    def __init__(self, num_classes, feat_dim, device, s=7.0, m=0.2):
        super(ArcLinear, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.device = device
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi-m)*m
        self.threshold = math.cos(math.pi-m)

    def forward(self, feat, label):
        eps = 1e-4
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        feat_l2norm = torch.div(feat, norms)
        feat_l2norm = feat_l2norm * self.s

        norms_w = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        weights_l2norm = torch.div(self.weights, norms_w)

        fc7 = torch.matmul(feat_l2norm, torch.transpose(weights_l2norm, 0, 1))

        if torch.cuda.is_available():
            label = label.cuda()
            fc7 = fc7.cuda()
        else:
            label = label.cpu()
            fc7 = fc7.cpu()

        target_one_hot = torch.zeros(len(label), 10).to(self.device)
        target_one_hot = target_one_hot.scatter_(1, label.unsqueeze(1), 1.)        
        zy = torch.addcmul(torch.zeros(fc7.size()).to(self.device), 1., fc7, target_one_hot)
        zy = zy.sum(-1)

        cos_theta = zy/self.s
        cos_theta = cos_theta.clamp(min=-1+eps, max=1-eps) # for numerical stability

        theta = torch.acos(cos_theta)
        theta = theta+self.m

        body = torch.cos(theta)
        new_zy = body*self.s

        diff = new_zy - zy
        diff = diff.unsqueeze(1)

        body = torch.addcmul(torch.zeros(diff.size()).to(self.device), 1., diff, target_one_hot)
        output = fc7+body

        return output.to(self.device)


class ArcLoss(nn.Module):
    """
    Additive Angular Margin loss module (ArcFace)
    Reference: https://arxiv.org/pdf/1801.07698.pdf
    
    :param nfeat: the number of features in the embedding
    :param nclass: the number of classes
    :param margin: the margin to separate classes in angular space
    :param s: the scaling factor for the feature vector. This will constrain
              the model to a hypersphere of radius s
    """
    
    def __init__(self, nfeat, nclass, margin=0.35, s=4.0):
        super(ArcLoss, self).__init__()
        self.nclass = nclass
        self.margin = margin
        self.s = s
        self.W = nn.Parameter(torch.Tensor(nfeat, nclass))
        self.loss_fn = nn.CrossEntropyLoss()
        self.last_output = torch.Tensor()
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
        # Calculate the logits, which will be our cosθj
        cos_theta_j = torch.mm(xnorm, Wnorm)
        # Get the cosθ corresponding to our classes
        cos_theta_yi = cos_theta_j.gather(1, y)
        # Get the angle separating x and W
        theta_yi = torch.acos(cos_theta_yi)
        # Apply the margin
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # One hot encoding for y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, y, 1.0)
        # Project margin differences into cosθj
        cos_theta_j += one_hot * (cos_theta_yi_margin - cos_theta_yi)
        # Apply the scaling
        cos_theta_j = self.s * cos_theta_j
        self.last_output = cos_theta_j
        # Apply softmax + cross entropy loss
        return self.loss_fn(cos_theta_j, y.view(-1))
    

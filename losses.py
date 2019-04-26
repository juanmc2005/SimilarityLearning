#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcLinear(nn.Module):
    """
    Additive Angular Margin loss module (ArcFace)
    Reference: https://arxiv.org/pdf/1801.07698.pdf
    
    :param nfeat: the number of features in the embedding
    :param nclass: the number of classes
    :param margin: the margin to separate classes in angular space
    :param s: the scaling factor for the feature vector. This will constrain
              the model to a hypersphere of radius s
    """
    
    def __init__(self, nfeat, nclass, margin=0.2, s=7.0):
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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss module
    Reference: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    
    :param margin: the margin to separate feature vectors considered different
    :param distance: the base distance to use (either 'euclidean' or 'cosine')
    """
    
    def __init__(self, distance, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(self, x1, x2, y):
        """
        Calculate the contrastive loss measure
        
        :param x1: a tensor
        :param x2: a tensor
        :param y: a non one-hot label tensor
        
        :return: the contrastive loss
        """
        # Calculate the distance between x1 and x2
        dist = self.distance(x1, x2)
        # Calculate the loss
        loss = (1-y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        # Return the mean loss for this batch
        return torch.sum(loss) / 2.0 / x1.size(0)

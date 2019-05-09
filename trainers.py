#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from  torch.utils.data import DataLoader
from losses.wrappers import LossWrapper
from losses.triplet import TripletLoss
from losses.center import SoftmaxCenterLoss
from losses.arcface import ArcLinear
from losses.contrastive import ContrastiveLoss
from models import ArcNet, ContrastiveNet, CenterNet
from distances import EuclideanDistance, CosineDistance, AccuracyCalculator
import visual


class BaseTrainer:
    
    def __init__(self, model, device, loss_fn, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_batch = len(self.train_loader)
        self.n_test_batch = len(self.test_loader)
        self.n_train = len(train_loader.dataset)
        self.n_test = len(test_loader.dataset)
        self.best_acc = 0
        
    def feed_forward(self, x, y):
        """
        FIXME this method could be removed. Feed forward differences between losses could be put in the models
        Compute the output of the model for a mini-batch. Return the embeddings and logits
        """
        raise NotImplementedError("The trainer must implement the method 'feed_forward'")
        
    def get_schedulers(self):
        """
        :return: a list of schedulers to use
        """
        raise NotImplementedError("The trainer must implement the method 'get_schedulers'")
        
    def get_optimizers(self):
        """
        :return: a list of optimizers to use
        """
        raise NotImplementedError("The trainer must implement the method 'get_optimizers'")
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        """
        FIXME all titles are pretty much the same, this method can be split in 2: get_title and get_params_str
        :return: a string representing the title for the plot of test embeddings with the best accuracy
        """
        raise NotImplementedError("The trainer must implement the method 'get_best_acc_plot_title'")
        
    def train(self, epochs=10, log_interval=20, train_accuracy=True):
        for i in range(1, epochs+1):
            self.train_epoch(i, log_interval, train_accuracy)
        
    def train_epoch(self, epoch, log_interval, train_accuracy):
        feat_train, y_train = [], []
        for sch in self.get_schedulers():
            sch.step()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Feed Forward
            feat, logits = self.feed_forward(x, y)
            loss = self.loss_fn(feat, logits, y)
            
            # Backprop
            optims = self.get_optimizers()
            for op in optims:
                op.zero_grad()
            loss.backward()
            for op in optims:
                op.step()
            
            # For accuracy tracking
            feat_train.append(feat)
            y_train.append(y)
            
            # Logging
            if i % log_interval == 0 or i == self.n_batch-1:
                print(f"Train Epoch: {epoch} [{100. * i / self.n_batch:.0f}%]\tLoss: {loss.item():.6f}")
        
        feat_train = torch.cat(feat_train, 0).float().detach().cpu().numpy()
        y_train = torch.cat(y_train, 0).detach().cpu().numpy()
        # FIXME cosine distance shouldn't be hardcoded here
        acc_calc = AccuracyCalculator(feat_train, y_train, CosineDistance())
        feat_test, y_test, test_correct, test_total = self.test(acc_calc, log_interval // 3)
        acc = 100 * test_correct / test_total
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"Test Accuracy: {test_correct} / {test_total} ({acc:.2f}%)")
        print("------------------------------------------------")
        if test_correct > self.best_acc:
            plot_name = f"test-feat-epoch-{epoch}"
            print(f"New Best Test Accuracy! Saving plot as {plot_name}")
            self.best_acc = test_correct
            visual.visualize(feat_test, y_test, self.get_best_acc_plot_title(epoch, acc), plot_name)
    
    def test(self, acc_calc, log_interval):
        correct, total = 0, 0
        feat_test, y_test = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Feed Forward
                feat, _ = self.feed_forward(x, y)
                feat = feat.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                
                # Track accuracy
                feat_test.append(feat)
                y_test.append(y)
                bcorrect, btotal = acc_calc.calculate_batch(feat, y)
                correct += bcorrect
                total += btotal
                
                # Logging
                if i % log_interval == 0 or i == self.n_test_batch-1:
                    print(f"Testing [{100. * i / self.n_test_batch:.0f}%]")
        return np.concatenate(feat_test), np.concatenate(y_test), correct, total


class ArcTrainer(BaseTrainer):

    def __init__(self, trainset, testset, device, nfeat, nclass, margin=0.2, s=7.0, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(ArcTrainer, self).__init__(
                ArcNet(),
                device,
                LossWrapper(nn.CrossEntropyLoss().to(device)),
                train_loader,
                test_loader)
        self.arc = ArcLinear(nfeat, nclass, margin, s).to(device)
        self.margin = margin
        self.s = s
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005),
                optim.SGD(self.arc.parameters(), lr=0.01)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 8, gamma=0.2),
                lr_scheduler.StepLR(self.optimizers[1], 8, gamma=0.2)
        ]
    
    def feed_forward(self, x, y):
        feat = self.model(x)
        logits = self.arc(feat, y)
        return feat, logits
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"ArcFace Loss (Epoch {epoch}) - {accuracy:.1f}% Accuracy - m={self.margin} s={self.s}"


class ContrastiveTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, margin=2.0, distance=EuclideanDistance(), batch_size=80):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)
        super(ContrastiveTrainer, self).__init__(
                ContrastiveNet(),
                device,
                ContrastiveLoss(device, margin, distance),
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
    
    def feed_forward(self, x, y):
        feat = self.model(x)
        return feat, feat
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Contrastive Loss (Epoch {epoch}) - {accuracy:.1f}% Accuracy - m={self.margin} - {self.distance}"


class SoftmaxTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(SoftmaxTrainer, self).__init__(
                CenterNet(),
                device,
                LossWrapper(nn.NLLLoss().to(device)),
                train_loader,
                test_loader)
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 10, gamma=0.5)
        ]
    
    def feed_forward(self, x, y):
        return self.model(x)
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Cross Entropy (Epoch {epoch}) - {accuracy:.1f}% Accuracy"


class TripletTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, margin=0.2, distance=EuclideanDistance(), batch_size=25):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)
        super(TripletTrainer, self).__init__(
                ContrastiveNet(),
                device,
                TripletLoss(device, margin, distance),
                train_loader,
                test_loader)
        self.margin = margin
        self.distance = distance
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 3, gamma=0.5)
        ]
    
    def feed_forward(self, x, y):
        feat = self.model(x)
        return feat, feat
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Triplet Loss (Epoch {epoch}) - {accuracy:.1f}% Accuracy - m={self.margin} - {self.distance}"


class CenterTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, nclass, loss_weight=1, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(CenterTrainer, self).__init__(
                CenterNet(),
                device,
                SoftmaxCenterLoss(device, nfeat, nclass, loss_weight),
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
    
    def feed_forward(self, x, y):
        return self.model(x)
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Center Loss (Epoch {epoch}) - {accuracy:.1f}% Accuracy - λ={self.loss_weight}"

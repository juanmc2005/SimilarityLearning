#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from losses.wrappers import LossWrapper
from losses.triplet import TripletLoss
from losses.center import SoftmaxCenterLoss
from losses.contrastive import ContrastiveLoss
from models import CommonNet, ArcNet, CenterNet
from distances import EuclideanDistance, CosineDistance, AccuracyCalculator
import visual


class BaseTrainer:
    
    def __init__(self, model, device, loss_fn, test_distance, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.test_distance = test_distance
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_batch = len(self.train_loader)
        self.n_test_batch = len(self.test_loader)
        self.n_train = len(train_loader.dataset)
        self.n_test = len(test_loader.dataset)
        self.best_acc = 0
        
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
        FIXME all titles are pretty much the same, this method can be split in 2: get_title (maybe __str__) and describe_params
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
            feat, logits = self.model(x, y)
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
        acc_calc = AccuracyCalculator(feat_train, y_train, self.test_distance)
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
                feat, _ = self.model(x, y)
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
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"ArcFace Loss (Epoch {epoch}) - {accuracy:.1f}% Accuracy - m={self.margin} s={self.s}"


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
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Contrastive Loss (Epoch {epoch}) - {accuracy:.1f}% Accuracy - m={self.margin} - {self.distance}"


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
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Cross Entropy (Epoch {epoch}) - {accuracy:.1f}% Accuracy"


class TripletTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, nfeat, margin=0.2, distance=EuclideanDistance(), batch_size=25):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)
        super(TripletTrainer, self).__init__(
                CommonNet(nfeat),
                device,
                TripletLoss(device, margin, distance),
                distance,
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
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Center Loss (Epoch {epoch}) - {accuracy:.1f}% Accuracy - λ={self.loss_weight}"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from  torch.utils.data import DataLoader
from CenterLoss import CenterLoss
from losses import ArcLinear, ContrastiveLoss, TripletLoss
from models import ArcNet, ContrastiveNet, CenterNet
from distances import EuclideanDistance
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
        self.accuracies = []
        self.best_acc = 0
        
    def feed_forward(self, x, y):
        """
        Compute the output of the model for a mini-batch and return the logits
        """
        raise NotImplementedError("The trainer must implement the method 'feed_forward'")
        
    def embed(self, x):
        """
        Compute the output of the model for a mini-batch and return the embeddings
        """
        raise NotImplementedError("The trainer must implement the method 'embed'")
        
    def get_schedulers(self):
        """
        Return the list of schedulers to use
        """
        raise NotImplementedError("The trainer must implement the method 'get_schedulers'")
        
    def get_optimizers(self):
        """
        Return the list of optimizers to use
        """
        raise NotImplementedError("The trainer must implement the method 'get_optimizers'")
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        """
        Return the desired title for the plot of test embeddings with the best accuracy
        """
        raise NotImplementedError("The trainer must implement the method 'get_best_acc_plot_title'")
    
    def batch_accuracy(self, logits, y):
        """
        Return a pair containing the amount of corrected predictions and the total of examples
        """
        raise NotImplementedError("The trainer must implement the method 'batch_accuracy'")
        
    def train(self, epochs=10, log_interval=20, train_accuracy=True):
        for i in range(1, epochs+1):
            self.train_epoch(i, log_interval, train_accuracy)
        
    def train_epoch(self, epoch, log_interval, train_accuracy):
        correct, total = 0, 0
        for sch in self.get_schedulers():
            sch.step()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Feed Forward
            logits = self.feed_forward(x, y)
            loss = self.loss_fn(logits, y)
            
            # Backprop
            optims = self.get_optimizers()
            for op in optims:
                op.zero_grad()
            loss.backward()
            for op in optims:
                op.step()
            
            # Accuracy tracking
            if train_accuracy:
                bcorrect, btotal = self.batch_accuracy(logits, y)
                correct += bcorrect
                total += btotal
            
            # Logging
            if i != 0 and i % log_interval == 0:
                print(f"Train Epoch: {epoch} [{100. * i / self.n_batch:.0f}%]\tLoss: {loss.item():.6f}")
        
        test_correct, test_total = self.test(log_interval // 3)
        acc = 100 * test_correct / test_total
        self.accuracies.append(acc)
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        if train_accuracy:
            print(f"Training Accuracy = {100 * correct / total:.0f}%")
        print(f"Test Accuracy: {test_correct} / {test_total} ({acc:.0f}%)")
        print("------------------------------------------------")
        if test_correct > self.best_acc:
            plot_name = f"test-feat-epoch-{epoch}"
            print(f"New Best Test Accuracy! Saving plot as {plot_name}")
            self.best_acc = test_correct
            self.visualize(self.test_loader, self.get_best_acc_plot_title(epoch, acc), plot_name)
    
    def test(self, log_interval):
        correct, total = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Feed Forward
                logits = self.feed_forward(x, y)
                
                # Track accuracy
                bcorrect, btotal = self.batch_accuracy(logits, y)
                correct += bcorrect
                total += btotal
                
                # Logging
                if i != 0 and i % log_interval == 0:
                    print(f"Testing [{100. * i / self.n_test_batch:.0f}%]")
        return correct, total
    
    def visualize(self, loader, title, filename):
        embs, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                embs.append(self.embed(x))
                targets.append(y)
        embs = torch.cat(embs, 0).float().data.cpu().numpy()
        targets = torch.cat(targets, 0).float().cpu().numpy()
        visual.visualize(embs, targets, title, filename)


class ArcTrainer(BaseTrainer):

    def __init__(self, trainset, testset, device, nfeat, nclass, margin=0.2, s=7.0, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(ArcTrainer, self).__init__(
                ArcNet(),
                device,
                nn.CrossEntropyLoss().to(device),
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
        return logits
        
    def embed(self, x):
        return self.model(x)
    
    def batch_accuracy(self, logits, y):
        _, predicted = torch.max(logits.data, 1)
        return (predicted == y.data).sum(), y.size(0)
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Test Embeddings (Epoch {epoch}) - {accuracy:.0f}% Accuracy - m={self.margin} s={self.s}"


class ContrastiveTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, margin=2.0, distance=EuclideanDistance(), batch_size=150):
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
                optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 10, gamma=0.5)
        ]
    
    def batch_accuracy(self, logits, y):
        return self.loss_fn.eval(logits, y)
    
    def feed_forward(self, x, y):
        return self.embed(x)
        
    def embed(self, x):
        return self.model(x)
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Test Embeddings (Epoch {epoch}) - {accuracy:.0f}% Accuracy - m={self.margin} - {self.distance}"


class SoftmaxTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(SoftmaxTrainer, self).__init__(
                CenterNet(),
                device,
                nn.NLLLoss(),
                train_loader,
                test_loader)
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 10, gamma=0.5)
        ]
    
    def batch_accuracy(self, logits, y):
        _, predicted = torch.max(logits.data, 1)
        return (predicted == y.data).sum(), y.size(0)
    
    def feed_forward(self, x, y):
        return self.model(x)[1]
        
    def embed(self, x):
        return self.model(x)[0]
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Test Embeddings (Epoch {epoch}) - {accuracy:.0f}% Accuracy"


class TripletTrainer(BaseTrainer):
    
    def __init__(self, trainset, testset, device, margin=0.2, distance=EuclideanDistance(), batch_size=50):
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
                lr_scheduler.StepLR(self.optimizers[0], 5, gamma=0.5)
        ]
    
    def batch_accuracy(self, logits, y):
        return self.loss_fn.eval(logits, y)
    
    def feed_forward(self, x, y):
        return self.embed(x)
        
    def embed(self, x):
        return self.model(x)
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Test Embeddings (Epoch {epoch}) - {accuracy:.0f}% Accuracy - m={self.margin} - {self.distance}"


class CenterTrainer:
    
    def __init__(self, model, device, loss_weight=1):
        self.loss_weight = loss_weight
        self.device = device
        self.model = model.to(device)
        self.nllloss = nn.NLLLoss().to(device)
        self.centerloss = CenterLoss(10, 2).to(device)
        self.optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        self.optimzer4center = optim.SGD(self.centerloss.parameters(), lr=0.5)
    
    def train(self, epoch, loader):
        print("Training... Epoch = %d" % epoch)
        ip1_loader = []
        idx_loader = []
        for i, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
    
            ip1, pred = self.model(data)
            loss = self.nllloss(pred, target) + self.loss_weight * self.centerloss(target, ip1)
    
            self.optimizer4nn.zero_grad()
            self.optimzer4center.zero_grad()
    
            loss.backward()
    
            self.optimizer4nn.step()
            self.optimzer4center.step()
    
            ip1_loader.append(ip1)
            idx_loader.append((target))
    
        feat = torch.cat(ip1_loader, 0)
        labels = torch.cat(idx_loader, 0)
        visual.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(),
                         "Epoch = {}".format(epoch), "epoch={}".format(epoch))

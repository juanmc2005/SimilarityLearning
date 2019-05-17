#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from distances import AccuracyCalculator
import visual


class TrainingConfig:
    """
    Abstracts the configuration of a Trainer.
    :param optimizers: a list of optimizers
    :param schedulers: a list of schedulers
    :param param_description: a string describing the relevant parameters of this trainer
    """
    def __init__(self, name, optimizers, schedulers, param_description=None):
        self.name = name
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.param_description = param_description


class TrainingListener:
    """
    A listener for the training process.
    """
    
    def on_before_train(self, n_batch):
        pass
    
    def on_before_epoch(self, epoch):
        pass
    
    def on_before_gradients(self, epoch, ibatch, feat, logits, y, loss):
        pass
    
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        pass
    
    def on_after_epoch(self, epoch):
        pass
    
    def on_before_test(self, n_batch):
        pass
    
    def on_batch_tested(self, ibatch, feat, correct, total):
        pass
    
    def on_after_test(self, n_batch):
        pass


class Optimizer(TrainingListener):
    
    def __init__(self, optimizers, schedulers):
        super(Optimizer, self).__init__()
        self.optimizers = optimizers
        self.schedulers = schedulers
    
    def on_before_epoch(self, epoch):
        for s in self.schedulers:
            s.step()
    
    def on_before_gradients(self, epoch, ibatch, feat, logits, y, loss):
        for o in self.optimizers:
            o.zero_grad()
    
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        for o in self.optimizers:
            o.step()


class Logger(TrainingListener):
    
    def __init__(self, interval):
        super(Logger, self).__init__()
        self.interval = interval
        self.last_log = None
        self.last_test_log = None
        self.n_batch = None
        self.n_test_batch = None
    
    def on_before_train(self, n_batch):
        self.n_batch = n_batch
    
    def on_before_epoch(self, epoch):
        self.last_log = -1
    
    def on_before_test(self, n_batch):
        self.n_test_batch = n_batch
        self.last_test_log = -1
    
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        progress = int(100. * (ibatch+1) / self.n_batch)
        if progress > self.last_log and progress % self.interval == 0:
            self.last_log = progress
            print(f"Train Epoch: {epoch} [{progress}%]\tLoss: {loss.item():.6f}")
    
    def on_batch_tested(self, ibatch, feat, correct, total):
        progress = int(100. * (ibatch+1) / self.n_test_batch)
        if progress > self.last_test_log and progress % self.interval == 0:
            self.last_test_log = progress
            print(f"Testing [{progress}%]")


class BaseTrainer:
    
    def __init__(self, model, device, loss_fn, test_distance, train_loader, test_loader, config, callbacks=[]):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.test_distance = test_distance
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.callbacks = callbacks
        self.callbacks.append(Optimizer(config.optimizers, config.schedulers))
        self.n_batch = len(self.train_loader)
        self.n_test_batch = len(self.test_loader)
        self.best_acc = 0
        
    def train(self, epochs):
        for cb in self.callbacks:
            cb.on_before_train(self.n_batch)
        for i in range(1, epochs+1):
            self.train_epoch(i)
        
    def train_epoch(self, epoch):
        self.model.train()
        feat_train, y_train = [], []
        for cb in self.callbacks:
            cb.on_before_epoch(epoch)
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Feed Forward
            feat, logits = self.model(x, y)
            loss = self.loss_fn(feat, logits, y)
            
            # Backprop
            for cb in self.callbacks:
                cb.on_before_gradients(epoch, i, feat, logits, y, loss)
                
            loss.backward()
            
            for cb in self.callbacks:
                cb.on_after_gradients(epoch, i, feat, logits, y, loss)
            
            # For accuracy tracking
            feat_train.append(feat)
            y_train.append(y)
        
        for cb in self.callbacks:
            cb.on_after_epoch(epoch)
        feat_train = torch.cat(feat_train, 0).float().detach().cpu().numpy()
        y_train = torch.cat(y_train, 0).detach().cpu().numpy()
        acc_calc = AccuracyCalculator(feat_train, y_train, self.test_distance)
        feat_test, y_test, test_correct, test_total = self.test(acc_calc)
        acc = 100 * test_correct / test_total
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"Test Accuracy: {test_correct} / {test_total} ({acc:.2f}%)")
        print("------------------------------------------------")
        if test_correct > self.best_acc:
            plot_name = f"test-feat-epoch-{epoch}"
            plot_title = f"{self.config.name} (Epoch {epoch}) - {acc:.1f}% Accuracy"
            if self.config.param_description is not None:
                plot_title += f" - {self.config.param_description}"
            print(f"New Best Test Accuracy! Saving plot as {plot_name}")
            self.best_acc = test_correct
            visual.visualize(feat_test, y_test, plot_title, plot_name)
    
    def test(self, acc_calc):
        self.model.eval()
        correct, total = 0, 0
        feat_test, y_test = [], []
        for cb in self.callbacks:
            cb.on_before_test(self.n_test_batch)
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
                
                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat, correct, total)
                
        for cb in self.callbacks:
            cb.on_after_test(self.n_test_batch)
        return np.concatenate(feat_test), np.concatenate(y_test), correct, total

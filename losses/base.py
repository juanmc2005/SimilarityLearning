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
    
    def on_after_epoch(self, epoch, model):
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


class Evaluator(TrainingListener):
    # FIXME split this enormous class
    # TODO visualization stuff should be a listener of this class
    # TODO logger should be a listener for this class. In that case, also simplify Logger class and TrainingListener
    def __init__(self, device, test_loader, distance, loss_name, param_desc=None, logger=None):
        self.device = device
        self.test_loader = test_loader
        self.distance = distance
        self.loss_name = loss_name
        self.param_desc = param_desc
        self.logger = logger
        self.feat_train, self.y_train = None, None
        self.best_acc = 0
        self.n_batch = len(test_loader)
    
    def _eval(self, model, acc_calc):
        model.eval()
        correct, total = 0, 0
        feat_test, y_test = [], []
        if self.logger is not None:
            self.logger.on_before_test(self.n_batch)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Feed Forward
                feat, _ = model(x, y)
                feat = feat.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                
                # Track accuracy
                feat_test.append(feat)
                y_test.append(y)
                bcorrect, btotal = acc_calc.calculate_batch(feat, y)
                correct += bcorrect
                total += btotal
                
                if self.logger is not None:
                    self.logger.on_batch_tested(i, feat, correct, total)
                
        if self.logger is not None:
            self.logger.on_after_test(self.n_batch)
        return np.concatenate(feat_test), np.concatenate(y_test), correct, total
    
    def on_before_epoch(self, epoch):
        self.feat_train, self.y_train = [], []
        
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.feat_train.append(feat)
        self.y_train.append(y)
    
    def on_after_epoch(self, epoch, model):
        feat_train = torch.cat(self.feat_train, 0).float().detach().cpu().numpy()
        y_train = torch.cat(self.y_train, 0).detach().cpu().numpy()
        acc_calc = AccuracyCalculator(feat_train, y_train, self.distance)
        feat_test, y_test, test_correct, test_total = self._eval(model, acc_calc)
        acc = 100 * test_correct / test_total
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"Test Accuracy: {test_correct} / {test_total} ({acc:.2f}%)")
        print("------------------------------------------------")
        if test_correct > self.best_acc:
            plot_name = f"test-feat-epoch-{epoch}"
            plot_title = f"{self.loss_name} (Epoch {epoch}) - {acc:.1f}% Accuracy"
            if self.param_desc is not None:
                plot_title += f" - {self.param_desc}"
            print(f"New Best Test Accuracy! Saving plot as {plot_name}")
            self.best_acc = test_correct
            visual.visualize(feat_test, y_test, plot_title, plot_name)


class BaseTrainer:
    
    def __init__(self, model, device, loss_fn, train_loader, callbacks=[]):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.callbacks = callbacks
        self.n_batch = len(self.train_loader)
        
    def train(self, epochs):
        for cb in self.callbacks:
            cb.on_before_train(self.n_batch)
        for i in range(1, epochs+1):
            self.train_epoch(i)
        
    def train_epoch(self, epoch):
        self.model.train()
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
        
        for cb in self.callbacks:
            cb.on_after_epoch(epoch, self.model)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from distances import AccuracyCalculator
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
    
    def describe_params(self):
        """
        :return: a string with the relevant parameter information for this trainer
        """
        return None
        
    def train(self, epochs, log_interval):
        for i in range(1, epochs+1):
            self.train_epoch(i, log_interval)
        
    def train_epoch(self, epoch, log_interval):
        last_log = -1
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
            progress = int(100. * (i+1) / self.n_batch)
            if progress > last_log and progress % log_interval == 0:
                last_log = progress
                print(f"Train Epoch: {epoch} [{progress}%]\tLoss: {loss.item():.6f}")
        
        feat_train = torch.cat(feat_train, 0).float().detach().cpu().numpy()
        y_train = torch.cat(y_train, 0).detach().cpu().numpy()
        acc_calc = AccuracyCalculator(feat_train, y_train, self.test_distance)
        feat_test, y_test, test_correct, test_total = self.test(acc_calc, log_interval)
        acc = 100 * test_correct / test_total
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"Test Accuracy: {test_correct} / {test_total} ({acc:.2f}%)")
        print("------------------------------------------------")
        if test_correct > self.best_acc:
            plot_name = f"test-feat-epoch-{epoch}"
            plot_title = f"{self} (Epoch {epoch}) - {acc:.1f}% Accuracy"
            desc = self.describe_params()
            if desc is not None:
                plot_title += f" - {desc}"
            print(f"New Best Test Accuracy! Saving plot as {plot_name}")
            self.best_acc = test_correct
            visual.visualize(feat_test, y_test, plot_title, plot_name)
    
    def test(self, acc_calc, log_interval):
        last_log = -1
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
                progress = int(100. * (i+1) / self.n_test_batch)
                if progress > last_log and progress % log_interval == 0:
                    last_log = progress
                    print(f"Testing [{progress}%]")
        return np.concatenate(feat_test), np.concatenate(y_test), correct, total

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models import SimNet
from datasets.base import SimDatasetPartition
from core.optim import Optimizer
import common


class TrainingListener:
    """
    A listener for the training cycle.
    """
    
    def on_before_train(self, checkpoint):
        pass
    
    def on_before_epoch(self, epoch):
        pass
    
    def on_before_gradients(self, epoch, ibatch, feat, logits, y, loss):
        pass
    
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        pass
    
    def on_after_epoch(self, epoch, model, loss_fn, optim):
        pass

    def on_after_train(self):
        pass
    

class TestListener:
    """
    A listener for the evaluation cycle
    """
    def on_before_test(self):
        pass
    
    def on_batch_tested(self, ibatch, feat):
        pass
    
    def on_after_test(self, epoch, feat_test, y_test, metric_value):
        pass
    
    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        pass


class Trainer:
    
    def __init__(self, loss_name: str, model: SimNet, loss_fn: nn.Module, partition: SimDatasetPartition,
                 optim: Optimizer, model_loader=None, callbacks: list = None):
        self.loss_name = loss_name
        self.model = model.to(common.DEVICE)
        self.loss_fn = loss_fn
        self.partition = partition
        self.optim = optim
        self.model_loader = model_loader
        self.callbacks = callbacks if callbacks is not None else []

    def _restore(self):
        if self.model_loader is not None:
            checkpoint = self.model_loader.restore(self.model, self.loss_fn, self.optim, self.loss_name)
            epoch = checkpoint['epoch']
            return checkpoint, epoch + 1
        else:
            return None, 1
        
    def train(self, epochs):
        checkpoint, epoch = self._restore()

        for cb in self.callbacks:
            cb.on_before_train(checkpoint)

        for i in range(epoch, epoch+epochs):
            self.train_epoch(i)

        for cb in self.callbacks:
            cb.on_after_train()
        
    def train_epoch(self, epoch):
        self.model.train()

        for cb in self.callbacks:
            cb.on_before_epoch(epoch)

        self.optim.scheduler_step()

        nbatches = self.partition.nbatches()
        for i in range(nbatches):
            x, y = next(self.partition)

            if isinstance(x, torch.Tensor):
                x = x.to(common.DEVICE)
            if isinstance(y, torch.Tensor):
                y = y.to(common.DEVICE)
            
            # Feed Forward
            feat, logits = self.model(x, y)
            loss = self.loss_fn(feat, logits, y)

            # Backprop
            for cb in self.callbacks:
                cb.on_before_gradients(epoch, i, feat, logits, y, loss)
                
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            for cb in self.callbacks:
                cb.on_after_gradients(epoch, i, feat, logits, y, loss.item())
        
        for cb in self.callbacks:
            cb.on_after_epoch(epoch, self.model, self.loss_fn, self.optim)

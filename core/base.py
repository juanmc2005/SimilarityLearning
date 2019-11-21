#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models import SimNet
from datasets.base import SimDatasetPartition
from core.optim import Optimizer
import common
import visual_utils as vis


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
                 optim: Optimizer, model_loader=None, callbacks: list = None, last_metric_fn=None):
        self.loss_name = loss_name
        self.model = model.to(common.DEVICE)
        self.loss_fn = loss_fn
        self.partition = partition
        self.optim = optim
        self.model_loader = model_loader
        self.callbacks = callbacks if callbacks is not None else []
        self.last_metric_fn = last_metric_fn if last_metric_fn is not None else lambda: None

    def _restore(self):
        if self.model_loader is not None:
            checkpoint = self.model_loader.restore(self.model, self.loss_fn, self.optim, self.loss_name)
            return checkpoint
        else:
            return None

    def _create_plots(self, exp_path: str, plots: list):
        print("Creating training plots before exiting...")
        for plot in plots:
            vis.visualize_logs(exp_path,
                               log_file_name=plot['log_file'],
                               metric_name=plot['metric'],
                               bottom=plot['bottom'],
                               top=plot['top'],
                               color=plot['color'],
                               title=plot['title'],
                               plot_file_name=plot['filename'])
        print("Done")

    def _start_training(self, epochs):
        checkpoint = self._restore()

        for cb in self.callbacks:
            cb.on_before_train(checkpoint)

        for i in range(1, epochs + 1):
            self.train_epoch(i)
            if self.optim.lrs()[0] < 1e-6:
                print('Stopping because LR dropped from 1e-6')
                break
            self.optim.scheduler_step(self.last_metric_fn())

        for cb in self.callbacks:
            cb.on_after_train()
        
    def train(self, epochs: int, exp_path: str, plots: list):
        try:
            self._start_training(epochs)
            print("Training finished")
            self._create_plots(exp_path, plots)
        except KeyboardInterrupt:
            print("\nStopped by user")
            self._create_plots(exp_path, plots)
        
    def train_epoch(self, epoch):
        self.model.train()

        for cb in self.callbacks:
            cb.on_before_epoch(epoch)

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

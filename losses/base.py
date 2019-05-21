#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from distances import AccuracyCalculator
import visual


class TrainingListener:
    """
    A listener for the training process.
    """
    
    def on_before_train(self):
        pass
    
    def on_before_epoch(self, epoch):
        pass
    
    def on_before_gradients(self, epoch, ibatch, feat, logits, y, loss):
        pass
    
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        pass
    
    def on_after_epoch(self, epoch, model, optim):
        pass
    

class TestListener:
    """
    A listener for the training process
    """
    def on_before_test(self):
        pass
    
    def on_batch_tested(self, ibatch, feat, correct, total):
        pass
    
    def on_after_test(self):
        pass
    
    def on_best_accuracy(self, epoch, model, optim, accuracy, feat, y):
        pass


class Optimizer:
    
    def __init__(self, optimizers, schedulers):
        super(Optimizer, self).__init__()
        self.OPTIM_KEY = 'optimizers'
        self.SCHED_KEY = 'schedulers'
        self.optimizers = optimizers
        self.schedulers = schedulers
        
    def state_dict(self):
        return {
                self.OPTIM_KEY: [op.state_dict() for op in self.optimizers],
                self.SCHED_KEY: [s.state_dict() for s in self.schedulers]
        }
    
    def load_state_dict(self, checkpoint):
        for i, op in enumerate(self.optimizers):
            op.load_state_dict(checkpoint[self.OPTIM_KEY][i])
        for i, s in enumerate(self.schedulers):
            s.load_state_dict(checkpoint[self.SCHED_KEY][i])
    
    def scheduler_step(self):
        for s in self.schedulers:
            s.step()
    
    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()
    
    def step(self):
        for o in self.optimizers:
            o.step()


class Logger:
    
    def __init__(self, interval, n_batch):
        super(Logger, self).__init__()
        self.interval = interval
        self.n_batch = n_batch
        self.train_log_ft = "Train Epoch: {epoch} [{progress}%]\tLoss: {loss:.6f}"
        self.test_log_ft = "Testing [{progress}%]"
        self.last_log = -1
    
    def _progress(self, i):
        progress = int(100. * (i+1) / self.n_batch)
        should_log = progress > self.last_log and progress % self.interval == 0
        return progress, should_log
    
    def restart(self):
        self.last_log = -1
        
    def on_train_batch(self, i, epoch, loss):
        progress, should_log = self._progress(i)
        if should_log:
            self.last_log = progress
            print(self.train_log_ft.format(epoch=epoch, progress=progress, loss=loss.item()))
    
    def on_test_batch(self, i):
        progress, should_log = self._progress(i)
        if should_log:
            self.last_log = progress
            print(self.test_log_ft.format(progress=progress))


class TrainLogger(TrainingListener):
    
    def __init__(self, interval, n_batch):
        super(TrainLogger, self).__init__()
        self.logger = Logger(interval, n_batch)
    
    def on_before_epoch(self, epoch):
        self.logger.restart()
    
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.logger.on_train_batch(ibatch, epoch, loss)


class TestLogger(TestListener):

    def __init__(self, interval, n_batch):
        super(TestLogger, self).__init__()
        self.logger = Logger(interval, n_batch)
    
    def on_before_test(self):
        self.logger.restart()
    
    def on_batch_tested(self, ibatch, feat, correct, total):
        self.logger.on_test_batch(ibatch)


class Visualizer(TestListener):
    
    def __init__(self, loss_name, param_desc=None):
        super(Visualizer, self).__init__()
        self.loss_name = loss_name
        self.param_desc = param_desc
    
    def on_best_accuracy(self, epoch, model, optim, accuracy, feat, y):
        plot_name = f"embeddings-epoch-{epoch}"
        plot_title = f"{self.loss_name} (Epoch {epoch}) - {accuracy:.1f}% Accuracy"
        if self.param_desc is not None:
            plot_title += f" - {self.param_desc}"
        print(f"Saving plot as {plot_name}")
        visual.visualize(feat, y, plot_title, plot_name)


class ModelSaver(TestListener):
    
    def __init__(self, path):
        super(ModelSaver, self).__init__()
        self.path = path
    
    def on_best_accuracy(self, epoch, model, optim, accuracy, feat, y):
        print(f"Saving model to {self.path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'accuracy': accuracy
        }, self.path)


class Evaluator(TrainingListener):
    
    def __init__(self, device, test_loader, distance, callbacks=[]):
        super(Evaluator, self).__init__()
        self.device = device
        self.test_loader = test_loader
        self.distance = distance
        self.callbacks = callbacks
        self.feat_train, self.y_train = None, None
        self.best_acc = 0
    
    def _eval(self, model, acc_calc):
        model.eval()
        correct, total = 0, 0
        feat_test, y_test = [], []
        for cb in self.callbacks:
            cb.on_before_test()
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
                
                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat, correct, total)
                
        for cb in self.callbacks:
            cb.on_after_test()
        return np.concatenate(feat_test), np.concatenate(y_test), correct, total
    
    def on_before_epoch(self, epoch):
        self.feat_train, self.y_train = [], []
        
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.feat_train.append(feat)
        self.y_train.append(y)
    
    def on_after_epoch(self, epoch, model, optim):
        feat_train = torch.cat(self.feat_train, 0).float().detach().cpu().numpy()
        y_train = torch.cat(self.y_train, 0).detach().cpu().numpy()
        acc_calc = AccuracyCalculator(feat_train, y_train, self.distance)
        feat_test, y_test, test_correct, test_total = self._eval(model, acc_calc)
        acc = 100 * test_correct / test_total
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"Test Accuracy: {test_correct} / {test_total} ({acc:.2f}%)")
        print("------------------------------------------------")
        if test_correct > self.best_acc:
            self.best_acc = test_correct
            print('New Best Test Accuracy!')
            for cb in self.callbacks:
                cb.on_best_accuracy(epoch, model, optim, acc, feat_test, y_test)


class BaseTrainer:
    
    def __init__(self, model, device, loss_fn, loader, optim, callbacks=[]):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.loader = loader
        self.optim = optim
        self.callbacks = callbacks
        
    def train(self, epochs):
        for cb in self.callbacks:
            cb.on_before_train()
        for i in range(1, epochs+1):
            self.train_epoch(i)
        
    def train_epoch(self, epoch):
        self.model.train()
        for cb in self.callbacks:
            cb.on_before_epoch(epoch)
        self.optim.scheduler_step()
        for i, (x, y) in enumerate(self.loader):
            x, y = x.to(self.device), y.to(self.device)
            
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
                cb.on_after_gradients(epoch, i, feat, logits, y, loss)
        
        for cb in self.callbacks:
            cb.on_after_epoch(epoch, self.model, self.optim)

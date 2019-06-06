#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import visual


class TrainingListener:
    """
    A listener for the training process.
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
    

class TestListener:
    """
    A listener for the training process
    """
    def on_before_test(self):
        pass
    
    def on_batch_tested(self, ibatch, feat):
        pass
    
    def on_after_test(self):
        pass
    
    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
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
        if self.OPTIM_KEY in checkpoint:
            for i, op in enumerate(self.optimizers):
                op.load_state_dict(checkpoint[self.OPTIM_KEY][i])
        if self.SCHED_KEY in checkpoint:
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
            print(self.train_log_ft.format(epoch=epoch, progress=progress, loss=loss))
    
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
    
    def on_batch_tested(self, ibatch, feat):
        self.logger.on_test_batch(ibatch)


class Visualizer(TestListener):
    
    def __init__(self, loss_name, param_desc=None):
        super(Visualizer, self).__init__()
        self.loss_name = loss_name
        self.param_desc = param_desc

    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        plot_name = f"embeddings-epoch-{epoch}"
        plot_title = f"{self.loss_name} (Epoch {epoch}) - {accuracy:.1f} Accuracy"
        if self.param_desc is not None:
            plot_title += f" - {self.param_desc}"
        print(f"Saving plot as {plot_name}")
        visual.visualize(feat, y, plot_title, plot_name)


class ModelSaver(TestListener):
    
    def __init__(self, loss_name, path):
        super(ModelSaver, self).__init__()
        self.loss_name = loss_name
        self.path = path
    
    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        print(f"Saving model to {self.path}")
        torch.save({
            'epoch': epoch,
            'trained_loss': self.loss_name,
            'common_state_dict': model.common_state_dict(),
            'loss_module_state_dict': model.loss_module.state_dict() if model.loss_module is not None else None,
            'loss_state_dict': loss_fn.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'accuracy': accuracy
        }, self.path)


class Evaluator(TrainingListener):
    
    def __init__(self, device, loader, metric, batch_transforms=[], callbacks=[]):
        super(Evaluator, self).__init__()
        self.device = device
        self.loader = loader
        self.metric = metric
        self.batch_transforms = batch_transforms
        self.callbacks = callbacks
        self.feat_train, self.y_train = None, None
        self.best_metric = 0
    
    def _eval(self, model):
        model.eval()
        feat_test, logits_test, y_test = [], [], []
        for cb in self.callbacks:
            cb.on_before_test()
        with torch.no_grad():
            for i in range(self.loader.nbatches()):
                x, y = next(self.loader)

                # Apply custom transformations to the batch before feeding the model
                for transform in self.batch_transforms:
                    x, y = transform(x, y)
                
                # Feed Forward
                feat, logits = model(x, y)
                feat = feat.detach().cpu().numpy()
                logits = logits.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                
                # Track accuracy
                feat_test.append(feat)
                logits_test.append(logits)
                y_test.append(y)
                self.metric.calculate_batch(feat, logits, y)
                
                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat)
                
        for cb in self.callbacks:
            cb.on_after_test()
        return np.concatenate(feat_test), np.concatenate(y_test)

    def on_before_train(self, checkpoint):
        if checkpoint is not None:
            self.best_metric = checkpoint['accuracy']
    
    def on_before_epoch(self, epoch):
        self.feat_train, self.y_train = [], []
        
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.feat_train.append(feat.float().detach().cpu().numpy())
        self.y_train.append(y.detach().cpu().numpy())
    
    def on_after_epoch(self, epoch, model, loss_fn, optim):
        feat_train = np.concatenate(self.feat_train)
        y_train = np.concatenate(self.y_train)
        self.metric.fit(feat_train, y_train)
        feat_test, y_test = self._eval(model)
        metric_value = self.metric.get()
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"Test Metric: {metric_value:.6f}")
        print("------------------------------------------------")
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            print('New Best Test Metric!')
            for cb in self.callbacks:
                cb.on_best_accuracy(epoch, model, loss_fn, optim, metric_value, feat_test, y_test)


class DeviceMapperTransform:

    def __init__(self, device):
        self.device = device

    def __call__(self, x, y):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        return x, y.to(self.device)


class BaseTrainer:
    
    def __init__(self, loss_name, model, device, loss_fn, loader, optim, recover=None, batch_transforms=[], callbacks=[]):
        self.loss_name = loss_name
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.loader = loader
        self.optim = optim
        self.recover = recover
        self.batch_transforms = batch_transforms
        self.callbacks = callbacks

    def _restore(self):
        if self.recover is not None:
            checkpoint = torch.load(self.recover)
            self.model.load_common_state_dict(checkpoint['common_state_dict'])
            if self.loss_name == checkpoint['trained_loss']:
                self.loss_fn.load_state_dict(checkpoint['loss_state_dict'])
                if self.model.loss_module is not None:
                    self.model.loss_module.load_state_dict(checkpoint['loss_module_state_dict'])
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
            epoch = checkpoint['epoch']
            accuracy = checkpoint['accuracy']
            print(f"Recovered Model. Epoch {epoch}. Test Metric {accuracy}")
            return checkpoint, epoch+1
        else:
            return None, 1
        
    def train(self, epochs):
        checkpoint, epoch = self._restore()
        for cb in self.callbacks:
            cb.on_before_train(checkpoint)
        for i in range(epoch, epoch+epochs+1):
            self.train_epoch(i)
        
    def train_epoch(self, epoch):
        self.model.train()

        for cb in self.callbacks:
            cb.on_before_epoch(epoch)

        self.optim.scheduler_step()

        for i in range(self.loader.nbatches()):
            x, y = next(self.loader)

            # Apply custom transformations to the batch before feeding the model
            for transform in self.batch_transforms:
                x, y = transform(x, y)
            
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import visual
from os.path import join
from models import SimNet
from datasets.base import SimDatasetPartition
import common


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

    def on_after_train(self):
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


class ScreenProgressLogger:
    
    def __init__(self, interval, n_batch):
        super(ScreenProgressLogger, self).__init__()
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
    
    def __init__(self, interval, nbatches, log_file_path: str):
        super(TrainLogger, self).__init__()
        self.nbatches = nbatches
        self.logger = ScreenProgressLogger(interval, nbatches)
        self.total_loss = 0
        self.log_file = open(log_file_path, 'w')
    
    def on_before_epoch(self, epoch):
        self.total_loss = 0
        self.logger.restart()
    
    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.total_loss += loss
        self.logger.on_train_batch(ibatch, epoch, loss)

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        mean_loss = self.total_loss / self.nbatches
        print(f"[Epoch {epoch} finished. Mean Loss: {mean_loss:.6f}]")
        self.log_file.write(str(mean_loss))

    def on_after_train(self):
        self.log_file.close()


class TestLogger(TestListener):

    def __init__(self, interval, n_batch):
        super(TestLogger, self).__init__()
        self.logger = ScreenProgressLogger(interval, n_batch)
    
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


class DeviceMapperTransform:

    def __init__(self, device):
        self.device = device

    def __call__(self, x, y):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        return x, y.to(self.device)


class ModelSaver:

    def __init__(self, loss_name: str):
        self.loss_name = loss_name

    def save(self, epoch: int, model: SimNet, loss_fn: nn.Module, optim: Optimizer, accuracy: float, filepath: str):
        print(f"Saving model to {filepath}")
        torch.save({
            'epoch': epoch,
            'trained_loss': self.loss_name,
            'common_state_dict': model.common_state_dict(),
            'loss_module_state_dict': model.loss_module.state_dict() if model.loss_module is not None else None,
            'loss_state_dict': loss_fn.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'accuracy': accuracy
        }, filepath)


class ModelLoader:

    def __init__(self, path: str):
        self.path = path

    def get_trained_loss(self):
        return torch.load(self.path)['trained_loss']

    def restore(self, model: SimNet, loss_fn: nn.Module, optimizer: Optimizer, current_loss: str):
        checkpoint = torch.load(self.path)
        model.load_common_state_dict(checkpoint['common_state_dict'])
        if current_loss == checkpoint['trained_loss']:
            loss_fn.load_state_dict(checkpoint['loss_state_dict'])
            if model.loss_module is not None:
                model.loss_module.load_state_dict(checkpoint['loss_module_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
        print(f"Recovered Model. Epoch {checkpoint['epoch']}. Test Metric {checkpoint['accuracy']}")
        return checkpoint

    def load(self, model: SimNet, current_loss: str):
        checkpoint = torch.load(self.path)
        model.load_common_state_dict(checkpoint['common_state_dict'])
        if current_loss == checkpoint['trained_loss'] and model.loss_module is not None:
            model.loss_module.load_state_dict(checkpoint['loss_module_state_dict'])
        return checkpoint


class BestModelSaver(TestListener):

    def __init__(self, task: str, loss_name: str, base_path: str):
        super(BestModelSaver, self).__init__()
        self.task = task
        self.loss_name = loss_name
        self.base_path = base_path
        self.saver = ModelSaver(loss_name)

    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        filepath = join(self.base_path, f"best-{self.task}-{self.loss_name}-epoch={epoch}-metric={accuracy:.3f}.pt")
        self.saver.save(epoch, model, loss_fn, optim, accuracy, filepath)


class RegularModelSaver(TrainingListener):

    def __init__(self, task: str, loss_name: str, base_path: str, interval: int):
        self.task = task
        self.loss_name = loss_name
        self.base_path = base_path
        self.interval = interval
        self.saver = ModelSaver(loss_name)

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        if epoch % self.interval == 0:
            filepath = join(self.base_path, f"{self.task}-{self.loss_name}-epoch={epoch}")
            self.saver.save(epoch, model, loss_fn, optim, 0, filepath)


class BaseTrainer:
    
    def __init__(self, loss_name: str, model: SimNet, loss_fn: nn.Module, partition: SimDatasetPartition,
                 optim: Optimizer, model_loader: ModelLoader = None,
                 batch_transforms: list = None, callbacks: list = None):
        self.loss_name = loss_name
        self.model = model.to(common.DEVICE)
        self.loss_fn = loss_fn
        self.partition = partition
        self.optim = optim
        self.model_loader = model_loader
        self.batch_transforms = batch_transforms if batch_transforms is not None else []
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

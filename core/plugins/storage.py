import torch
import torch.nn as nn
from os.path import join
import core.base as base
from core.optim import Optimizer
from models import SimNet
import common


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

    def __init__(self, path: str, restore_optimizer: bool = True):
        self.path = path
        self.restore_optimizer = restore_optimizer

    def get_trained_loss(self):
        return torch.load(self.path, map_location=common.DEVICE)['trained_loss']

    def restore(self, model: SimNet, loss_fn: nn.Module, optimizer: Optimizer, current_loss: str):
        checkpoint = torch.load(self.path, map_location=common.DEVICE)
        model.load_common_state_dict(checkpoint['common_state_dict'])
        if current_loss == checkpoint['trained_loss']:
            loss_fn.load_state_dict(checkpoint['loss_state_dict'])
            if model.loss_module is not None:
                model.loss_module.load_state_dict(checkpoint['loss_module_state_dict'])
            if self.restore_optimizer:
                optimizer.load_state_dict(checkpoint['optim_state_dict'])
        print(f"Recovered Model. Epoch {checkpoint['epoch']}. Dev Metric {checkpoint['accuracy']}")
        return checkpoint

    def load(self, model: SimNet, current_loss: str):
        checkpoint = torch.load(self.path, map_location=common.DEVICE)
        model.load_common_state_dict(checkpoint['common_state_dict'])
        if current_loss == checkpoint['trained_loss'] and model.loss_module is not None:
            model.loss_module.load_state_dict(checkpoint['loss_module_state_dict'])
        return checkpoint


class BestModelSaver(base.TestListener):

    def __init__(self, task: str, loss_name: str, base_path: str, experience_name: str):
        super(BestModelSaver, self).__init__()
        self.task = task
        self.loss_name = loss_name
        self.base_path = base_path
        self.experience_name = experience_name
        self.saver = ModelSaver(loss_name)

    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        filepath = join(
            self.base_path,
            f"{self.experience_name}-best-{self.task}-{self.loss_name}-epoch={epoch}-metric={accuracy:.3f}.pt")
        self.saver.save(epoch, model, loss_fn, optim, accuracy, filepath)


class RegularModelSaver(base.TrainingListener):

    def __init__(self, task: str, loss_name: str, base_path: str, interval: int, experience_name: str):
        self.task = task
        self.loss_name = loss_name
        self.base_path = base_path
        self.interval = interval
        self.experience_name = experience_name
        self.saver = ModelSaver(loss_name)

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        if epoch % self.interval == 0:
            filepath = join(self.base_path, f"{self.experience_name}-{self.task}-{self.loss_name}-epoch={epoch}.pt")
            self.saver.save(epoch, model, loss_fn, optim, 0, filepath)
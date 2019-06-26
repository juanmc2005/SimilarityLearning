import os
from os.path import isdir
import torch
import numpy as np
import random
import losses.config as cf
from distances import CosineDistance, EuclideanDistance


"""
A string to explain all possible loss names
"""
LOSS_OPTIONS_STR = 'softmax / contrastive / triplet / arcface / center / coco'

"""
Common seed for every script
"""
SEED = 124

"""
PyTorch device: GPU if available, CPU otherwise
"""
_use_cuda = torch.cuda.is_available() and True
# TODO This device is a constant for everyone, we can remove the 'device' parameters and use this directly
# TODO or maybe create a singleton object
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')


def set_custom_seed(seed: int = None):
    """
    Set the same seed for all sources of random computations
    :return: nothing
    """
    if seed is None:
        seed = SEED
    print(f"[Seed: {seed}]")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def enabled_str(value: bool) -> str:
    """
    :param value: a boolean
    :return: ENABLED if value is True, DISABLED otherwise
    """
    return 'ENABLED' if value else 'DISABLED'


def create_log_dir(exp_id: str, task: str, loss: str):
    """
    Create the directory where logs, models, plots and other experiment related files will be stored
    :param exp_id: the name for this experiment
    :param task: the name of the task
    :param loss: the name of the loss function that will be optimized
    :return: the name of the created directory, or exit the program if the directory exists
    """
    log_path = f"tmp/{exp_id}-{task}-{loss}"
    if isdir(log_path):
        print(f"The experience directory '{log_path}' already exists")
        exit(1)
    os.mkdir(log_path)
    return log_path


def get_config(loss: str, nfeat: int, nclass: int, task: str, margin: float) -> cf.LossConfig:
    """
    Create a loss configuration object based on parameteres given by the user
    :param loss: the loss function name
    :param nfeat: the dimension of the embeddings
    :param nclass: the number of classes
    :param task: the task for which the loss will be used
    :param margin: a margin to use in contrastive, triplet and arcface losses
    :return: a loss configuration object
    """
    if loss == 'softmax':
        return cf.SoftmaxConfig(DEVICE, nfeat, nclass)
    elif loss == 'contrastive':
        print(f"[Margin: {margin}]")
        return cf.ContrastiveConfig(DEVICE,
                                    margin=margin,
                                    distance=CosineDistance(),
                                    size_average=False,
                                    online=task != 'sts')
    elif loss == 'triplet':
        print(f"[Margin: {margin}]")
        return cf.TripletConfig(DEVICE,
                                margin=margin,
                                distance=EuclideanDistance(),
                                size_average=False,
                                online=task != 'sts')
    elif loss == 'arcface':
        print(f"[Margin: {margin}]")
        return cf.ArcFaceConfig(DEVICE, nfeat, nclass, margin=margin)
    elif loss == 'center':
        return cf.CenterConfig(DEVICE, nfeat, nclass, distance=CosineDistance())
    elif loss == 'coco':
        return cf.CocoConfig(DEVICE, nfeat, nclass)
    elif loss == 'kldiv':
        return cf.KLDivergenceConfig(DEVICE, nfeat)
    else:
        raise ValueError(f"Loss function should be one of: {LOSS_OPTIONS_STR}")

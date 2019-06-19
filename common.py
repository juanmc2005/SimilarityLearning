import torch
import numpy as np
import random
import losses.config as cf
from distances import CosineDistance


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


def set_custom_seed():
    """
    Set the same seed for all sources of random computations
    :return: nothing
    """
    print(f"[Seed: {SEED}]")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def enabled_str(value: bool) -> str:
    """
    :param value: a boolean
    :return: ENABLED if value is True, DISABLED otherwise
    """
    return 'ENABLED' if value else 'DISABLED'


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
        return cf.TripletConfig(DEVICE, margin=margin)
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

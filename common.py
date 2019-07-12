import argparse
import time
import os
from os.path import isdir
import torch
import numpy as np
import random

import losses.config as cf
from losses.triplet import SemiHardNegative, BatchAll, HardestNegative, HardestPositiveNegative, TripletSamplingStrategy
from distances import CosineDistance


"""
A string to explain all possible loss names
"""
LOSS_OPTIONS_STR = 'softmax / contrastive / triplet / arcface / center / coco'
TRIPLET_SAMPLING_OPTIONS_STR = 'all / semihard-neg / hardest-neg / hardest-pos-neg'

"""
Common seed for every script
"""
SEED = 124

"""
PyTorch device: GPU if available, CPU otherwise
"""
_use_cuda = torch.cuda.is_available() and True
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


def get_config(loss: str, nfeat: int, nclass: int, task: str, margin: float,
               triplet_strategy: str, semihard_n: int = 10) -> cf.LossConfig:
    """
    Create a loss configuration object based on parameteres given by the user
    :param loss: the loss function name
    :param nfeat: the dimension of the embeddings
    :param nclass: the number of classes
    :param task: the task for which the loss will be used
    :param margin: a margin to use in contrastive, triplet and arcface losses
    :param triplet_strategy: The name of the triplet sampling strategy as received via script arguments
    :param semihard_n: the number of negatives to keep when using a semi-hard negative triplet sampling strategy
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
                                distance=CosineDistance(),
                                size_average=task != 'sts',
                                online=task != 'sts',
                                sampling=get_triplet_strategy(triplet_strategy, semihard_n))
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


def get_triplet_strategy(strategy: str, semihard_n: int) -> TripletSamplingStrategy:
    """
    Create a triplet sampling strategy object based on a script argument
    :param strategy: the name of the strategy received as a script argument
    :param semihard_n: the number of negatives to keep when using a semi-hard negative strategy
    :return: a TripletSamplingStrategy object
    """
    if strategy == 'all':
        return BatchAll()
    elif strategy == 'semihard-neg':
        return SemiHardNegative(semihard_n)
    elif strategy == 'hardest-neg':
        return HardestNegative()
    elif strategy == 'hardest-pos-neg':
        return HardestPositiveNegative()
    else:
        raise ValueError(f"Triplet strategy should be one of: {TRIPLET_SAMPLING_OPTIONS_STR}")


def get_arg_parser():
    """
    :return: an ArgumentParser with the common configuration for all tasks
    """
    launch_datetime = time.strftime('%c')
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, help=LOSS_OPTIONS_STR)
    parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Steps (in percentage) to show epoch progress. Default value: 10')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
    parser.add_argument('--plot', dest='plot', action='store_true',
                        help='Plot distance distribution for same and different speakers')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help='Do NOT plot distance distribution for same and different speakers')
    parser.set_defaults(plot=True)
    parser.add_argument('--save', dest='save', action='store_true', help='Save best accuracy models')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do NOT save best accuracy models')
    parser.set_defaults(save=True)
    parser.add_argument('--recover', type=str, default=None, help='The path to the saved model to recover for training')
    parser.add_argument('-m', '--margin', type=float, default=2., help='The margin to use for the losses that need it')
    parser.add_argument('--exp-id', type=str, default=f"EXP-{launch_datetime.replace(' ', '-')}",
                        help='An identifier for the experience')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--triplet-strategy', type=str, default='all',
                        help=F'Triplet sampling strategy. Possible values: {TRIPLET_SAMPLING_OPTIONS_STR}')
    parser.add_argument('--semihard-negatives', type=int, default=10,
                        help='The number of negatives to keep when using semi-hard negative triplet sampling strategy')
    parser.add_argument('--recover-optim', dest='recover_optim', action='store_true',
                        help='Recover optimizer state from model checkpoint')
    parser.add_argument('--no-recover-optim', dest='recover_optim', action='store_false',
                        help='Do NOT recover optimizer state from model checkpoint')
    parser.set_defaults(recover_optim=True)
    return parser


def dump_params(filepath: str, args):
    with open(filepath, 'w') as out:
        for k, v in sorted(vars(args).items()):
            out.write(f"{k}={v}\n")


def get_basic_plots(lr: float, batch_size: int, eval_metric: str, eval_metric_color: str) -> list:
    return [
        {
            'log_file': 'loss.log',
            'metric': 'Loss',
            'color': 'blue',
            'title': f'Train Loss - lr={lr} - batch_size={batch_size}',
            'filename': 'loss-plot'
        },
        {
            'log_file': 'metric.log',
            'metric': eval_metric,
            'color': eval_metric_color,
            'title': f'Dev {eval_metric} - lr={lr} - batch_size={batch_size}',
            'filename': f"dev-{eval_metric.lower().replace(' ', '-')}-plot"
        }
    ]

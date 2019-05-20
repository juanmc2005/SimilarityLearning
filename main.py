import torch
import argparse
from distances import CosineDistance
from losses.base import BaseTrainer, TrainLogger, TestLogger, Evaluator, Visualizer
from losses import config as cf
from datasets import mnist


# Constants and script arguments
loss_options = 'softmax / contrastive / triplet / arcface / center / coco'
use_cuda = torch.cuda.is_available() and True
nfeat, nclass = 2, 10
seed = 999
device = torch.device('cuda' if use_cuda else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=str, help='Path to MNIST dataset')
parser.add_argument('--loss', type=str, help=loss_options)
parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')
parser.add_argument('-c', '--controlled', type=bool, default=True, help='Whether to set a fixed seed to control the training environment. Default value: True')
parser.add_argument('--log-interval', type=int, default=10, help='Steps (in percentage) to show epoch progress. Default value: 10')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')


def get_config(loss):
    if loss == 'softmax':
        return cf.softmax(device, nfeat, nclass)
    elif loss == 'contrastive':
        return cf.contrastive(device, nfeat)
    elif loss == 'triplet':
        return cf.triplet(device, nfeat)
    elif loss == 'arcface':
        return cf.arcface(device, nfeat, nclass)
    elif loss == 'center':
        return cf.center(device, nfeat, nclass, distance=CosineDistance())
    elif loss == 'coco':
        return cf.coco(device, nfeat, nclass)
    else:
        raise ValueError(f"Loss function should be one of: {loss_options}")


# Parse arguments and set custom seed if requested
args = parser.parse_args()
if args.controlled:
    print(f"Training with seed: {seed}")
    torch.manual_seed(seed)

# Load Dataset
train_loader, test_loader = mnist(args.mnist, args.batch_size)

# Get loss dependent configuration
config = get_config(args.loss)

# Create trainer with plugins
test_callbacks = [
        TestLogger(args.log_interval, len(test_loader)),
        Visualizer(config['name'], config['param_desc'])
]
trainer = BaseTrainer(config['model'], device, config['loss'], train_loader, callbacks=[
        TrainLogger(args.log_interval, len(train_loader)),
        config['optim'],
        Evaluator(device, test_loader, config['test_distance'], test_callbacks)
])

# Start training
trainer.train(args.epochs)

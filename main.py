import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from distances import CosineDistance
from losses.base import BaseTrainer, TrainLogger, TestLogger, Evaluator
from losses import config as cf


# Constants and Config
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


# Init
args = parser.parse_args()
if args.controlled:
    print(f"Training with seed: {seed}")
    torch.manual_seed(seed)

# Load Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(args.mnist, download=True, train=True, transform=transform)
testset = datasets.MNIST(args.mnist, download=True, train=False, transform=transform)
train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=4)

# Train
config = get_config(args.loss)
test_logger = TestLogger(args.log_interval, len(test_loader))
trainer = BaseTrainer(config['model'], device, config['loss'], train_loader, callbacks=[
        TrainLogger(args.log_interval, len(train_loader)),
        config['optim'],
        Evaluator(device, test_loader, config['test_distance'],
                  config['name'], config['param_desc'], callbacks=[test_logger])
])

trainer.train(args.epochs)

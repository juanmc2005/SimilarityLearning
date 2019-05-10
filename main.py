import torch
from torchvision import datasets, transforms
from distances import CosineDistance
from losses.arcface import ArcTrainer
from losses.contrastive import ContrastiveTrainer
from losses.triplet import TripletTrainer
from losses.softmax import SoftmaxTrainer
from losses.center import CenterTrainer
from losses.coco import CocoTrainer
import argparse


# Constants and Config
loss_options = 'softmax / contrastive / triplet / arcface / center / coco'
use_cuda = torch.cuda.is_available() and True
nfeat, nclass = 2, 10
device = torch.device('cuda' if use_cuda else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=str, help='Path to MNIST dataset')
parser.add_argument('--loss', type=str, help=loss_options)
parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')


def get_trainer(loss):
    if loss == 'softmax':
        return SoftmaxTrainer(trainset, testset, device, nfeat, nclass)
    elif loss == 'contrastive':
        return ContrastiveTrainer(trainset, testset, device, nfeat, margin=0.2, distance=CosineDistance())
    elif loss == 'triplet':
        return TripletTrainer(trainset, testset, device, nfeat, margin=2.0)
    elif loss == 'arcface':
        return ArcTrainer(trainset, testset, device, nfeat, nclass)
    elif loss == 'center':
        return CenterTrainer(trainset, testset, device, nfeat, nclass)
    elif loss == 'coco':
        return CocoTrainer(trainset, testset, device, nfeat, nclass)
    else:
        raise ValueError(f"Loss function should be one of: {loss_options}")


# Init
args = parser.parse_args()
torch.manual_seed(999)

# Load Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(args.mnist, download=True, train=True, transform=transform)
testset = datasets.MNIST(args.mnist, download=True, train=False, transform=transform)

# Train
trainer = get_trainer(args.loss)
trainer.train(args.epochs, log_interval=30, train_accuracy=False)

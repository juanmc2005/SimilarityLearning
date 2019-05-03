import torch
from torchvision import datasets, transforms
from trainers import ContrastiveTrainer, ArcTrainer, SoftmaxTrainer, TripletTrainer
from distances import CosineDistance
import argparse


def get_trainer(loss):
    if loss == 'softmax':
        return SoftmaxTrainer(trainset, testset, device)
    elif loss == 'contrastive':
        return ContrastiveTrainer(trainset, testset, device, margin=0.3, distance=CosineDistance())
    elif loss == 'triplet':
        return TripletTrainer(trainset, testset, device, margin=0.3, distance=CosineDistance())
    elif loss == 'arcface':
        return ArcTrainer(trainset, testset, device, nfeat=2, nclass=10)
    else:
        raise ValueError('Loss function should be one of: softmax / contrastive / triplet / arcface')


# Config
use_cuda = torch.cuda.is_available() and True
device = torch.device('cuda' if use_cuda else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=str, help='Path to MNIST dataset')
parser.add_argument('--loss', type=str, help='softmax / contrastive / triplet / arcface')
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
trainer.train(epochs=15, log_interval=60, train_accuracy=False)


import torch
from torchvision import datasets, transforms
from distances import CosineDistance
import trainers as tr
import argparse


# Constants and Config
loss_options = 'softmax / contrastive / triplet / arcface / center'
use_cuda = torch.cuda.is_available() and True
device = torch.device('cuda' if use_cuda else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=str, help='Path to MNIST dataset')
parser.add_argument('--loss', type=str, help=loss_options)
parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')


def get_trainer(loss):
    if loss == 'softmax':
        return tr.SoftmaxTrainer(trainset, testset, device)
    elif loss == 'contrastive':
        return tr.ContrastiveTrainer(trainset, testset, device, margin=0.2, distance=CosineDistance())
    elif loss == 'triplet':
        return tr.TripletTrainer(trainset, testset, device, margin=0.2, distance=CosineDistance())
    elif loss == 'arcface':
        return tr.ArcTrainer(trainset, testset, device, nfeat=2, nclass=10)
    elif loss == 'center':
        return tr.CenterTrainer(trainset, testset, device, nfeat=2, nclass=10)
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


import torch
from torchvision import datasets, transforms
from trainers import ContrastiveTrainer, ArcTrainer, SoftmaxTrainer
from distances import CosineDistance
import argparse

def arc_trainer():
    return ArcTrainer(trainset, testset, device, nfeat=2, nclass=10)

def contrastive_trainer():
    return ContrastiveTrainer(trainset, testset, device, margin=0.3, distance=CosineDistance())

def softmax_trainer():
    return SoftmaxTrainer(trainset, testset, device)

# Config
use_cuda = torch.cuda.is_available() and True
device = torch.device('cuda' if use_cuda else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=str, help='Path to MNIST dataset')
args = parser.parse_args()
torch.manual_seed(999)

# Load Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(args.mnist, download=True, train=True, transform=transform)
testset = datasets.MNIST(args.mnist, download=True, train=False, transform=transform)

# Train
trainer = contrastive_trainer()
trainer.train(epochs=50, log_interval=40, train_accuracy=False)


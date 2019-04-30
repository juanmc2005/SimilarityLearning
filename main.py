import torch
from torchvision import datasets, transforms
from trainers import ContrastiveTrainer, ArcTrainer, SoftmaxTrainer

def arc_trainer():
    return ArcTrainer(trainset, testset, device, nfeat=2, nclass=10)

def contrastive_trainer():
    return ContrastiveTrainer(trainset, testset, device, margin=2.0)

def softmax_trainer():
    return SoftmaxTrainer(trainset, testset, device)

# Config
use_cuda = torch.cuda.is_available() and True
device = torch.device('cuda' if use_cuda else 'cpu')
mnist_path = '/localHD/MNIST' if use_cuda else '../MNIST'
torch.manual_seed(999)

# Load Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(mnist_path, download=True, train=True, transform=transform)
testset = datasets.MNIST(mnist_path, download=True, train=False, transform=transform)

# Train
trainer = arc_trainer()
trainer.train(epochs=50, log_interval=40, train_accuracy=True)


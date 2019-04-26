import torch
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
from models import ContrastiveNet, ArcNet
from trainers import ContrastiveTrainer, ArcTrainer, ArcTrainerBetter
from datasets import ContrastiveDataset

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


def contrastive():
    # Prepare Dataset
    # TODO shuffle and recombine train and test before each epoch
    print("Recombining Dataset...")
    xtrain = trainset.data.unsqueeze(1).float()
    ytrain = trainset.targets
    xtest = testset.data.unsqueeze(1).float()
    ytest = testset.targets
    dataset = ContrastiveDataset(xtrain, ytrain)
    test_dataset = ContrastiveDataset(xtest, ytest)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    trainer = ContrastiveTrainer(ContrastiveNet(), device, margin=2.0) # euclidean by default
    #trainer = ContrastiveTrainer(ContrastiveNet(), device, margin=0.3, distance='cosine')
    return trainer, loader, test_loader


def arc():
    trainer = ArcTrainer(ArcNet(), device, nfeat=2, nclass=10)
    loader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
    return trainer, loader, test_loader


#train_sample = [trainset[i] for i in range(5000)]
#test_sample = [testset[i] for i in range(5000)]
#trainer = ArcTrainerBetter(train_sample, test_sample, device, nfeat=2, nclass=10)
#trainer.train(epochs=15, log_interval=10)

trainer, train_loader, test_loader = contrastive()
visu_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
for epoch in range(2):
    trainer.train(epoch+1, train_loader, test_loader, visu_loader)


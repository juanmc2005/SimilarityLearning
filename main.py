import torch
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
from models import Net
from trainers import ContrastiveTrainer
from datasets import ContrastiveDataset

# Config
use_cuda = torch.cuda.is_available() and True
device = torch.device('cuda' if use_cuda else 'cpu')
mnist_path = '/localHD/MNIST' if use_cuda else '../MNIST'

# Load Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(mnist_path, download=True, train=True ,transform=transform)
testset = datasets.MNIST(mnist_path, download=True, train=False, transform=transform)

# Prepare Dataset
print("Recombining Dataset...")
xtrain = trainset.data.unsqueeze(1).type(torch.FloatTensor)
ytrain = trainset.targets
xtest = testset.data.unsqueeze(1).type(torch.FloatTensor)
ytest = testset.targets
dataset = ContrastiveDataset(xtrain, ytrain)
test_dataset = ContrastiveDataset(xtest, ytest)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
#xvis = trainset.data[:1000].unsqueeze(1).type(torch.FloatTensor).to(device)
#yvis = trainset.targets.data[:1000]

# Model
model = Net()

# Training
trainer = ContrastiveTrainer(model, device, margin=0.5, distance='cosine')
#visualize(model(xvis).data.cpu().numpy(), yvis.cpu().numpy(), 0)
for epoch in range(30):
    trainer.train(epoch+1, loader, test_loader)


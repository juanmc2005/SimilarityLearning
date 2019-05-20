# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def mnist(path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
    testset = datasets.MNIST(path, download=True, train=False, transform=transform)
    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

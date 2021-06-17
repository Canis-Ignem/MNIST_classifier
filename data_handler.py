import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_data(batch_size=64):
    
    transform = transforms.Compose([transforms.ToTensor()])

    # training data and trainloader
    train_data = MNIST('', download=True, train=True, transform=transform)
    test_data = MNIST('', download=True, train=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    # x_train, y_train = next(iter(train_loader))
    # x_train = x_train.view(-1, 1, 784)  # x_train shape [64, 1, 784]
    # x_val, y_val = next(iter(test_loader))
    # x_val = x_train.view(-1, 1, 784)  # x_val shape [64, 1, 784]

    return train_loader , test_loader 



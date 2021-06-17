import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

pth = ''

def get_data(batch_size=32):
    
    transform = transforms.Compose([transforms.ToTensor()])

    #training data and trainloader
    train_data = MNIST('', download=True, train=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)

    return train_dtrain_loaderata

x = get_data( )

print(x.shape)

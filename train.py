import numpy as np
import torch
import torch.nn as nn
import torchsummary as summary

# from model import model
# import data_handler as dh

import torch.optim as optim

from sklearn.metrics import accuracy_score



# x_train, x_test, y_train, y_test = 

train_loss = []
val_loss = []
accuracy = []




n_epochs = 10
print_every = 40

def train(model, optimizer, criterion, epochs, x_train, x_test, y_train, y_test):
    for i in range(epochs):


        for x, y in zip(x_train, y_train):

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                model = model.cuda()

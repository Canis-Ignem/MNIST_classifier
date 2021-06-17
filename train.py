import numpy as np
import torch
import torch.nn as nn
import torchsummary as summary

from linear_model import model
import data_handler as dh

import torch.optim as optim

from sklearn.metrics import accuracy_score




train_loss = []
val_loss = []
accuracy = []

optimizer = optim.Adam(model.parameters(), lr=0.01)
#scheduler = ExponentialLR(optimizer, gamma=0.9)
# scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1) 
criterion = nn.CrossEntropyLoss()


n_epochs = 11
print_every = 2

def train(model, optimizer, criterion, epochs):
    for i in range(epochs):

        train, test = dh.get_data()

        running_loss = 0
        for x, y in train:

            x = x.reshape(-1,784)
            
            if torch.cuda.is_available():
                x = x.cuda()
                
                y = y.cuda()
                #print(y)
                model = model.cuda()
            # else:
            #     x = x
            #     y = y

                
          
            optimizer.zero_grad()

            output = model.forward(x)

            # print(output.shape)
            # print(y.shape)
            
            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % print_every == 0:
                print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
                running_loss = 0


    torch.save(model, 'model.pth')    

train(model, optimizer, criterion, n_epochs)



            

            



        
    

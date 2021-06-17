import numpy as np
import torch
import torch.nn as nn
import torchsummary as summary

from linear_model import model
import data_handler as dh

import torch.optim as optim
from matplotlib import pyplot as plt 
from sklearn.metrics import accuracy_score


best_val = 10
best_acc = 0
train_loss = []
val_loss = []
accuracy = []

optimizer = optim.Adam(model.parameters(), lr=0.01) # 0.005
#scheduler = ExponentialLR(optimizer, gamma=0.9)
# scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1) 
criterion = nn.CrossEntropyLoss()

n_batches = 938
n_epochs = 2
print_every = 50

def train(model, optimizer, criterion, epochs):
    for i in range(epochs):

        train, test = dh.get_data()
        #print(len(train))
        #print(len(t))
        running_loss = 0
        for j,(x, y) in enumerate(train):

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

            if j % print_every == 0 and j != 0:
                

                correct_test = 0
                total_test = 0
                validation_loss = 0

                for k, (x,y) in enumerate(test):
                    if k == 50:
                        break
                    with torch.no_grad(): 
                    
                        x = x.reshape(-1,784).cuda()
                        y = y.cuda()

                        output = model(x)
                        validation_loss += criterion(output,y)

                    y = y.detach().cpu()
                    for idx, i in enumerate(output):
                        if torch.argmax(i) == y[idx]:
                            correct_test += 1
                        total_test += 1

                #print(correct_test/total_test)
                train_loss.append(running_loss/print_every)
                val_loss.append(validation_loss/n_batches)
                accuracy.append(correct_test/total_test)
                if best_val > val_loss/n_batches:
                    best_val = val_loss/n_batches
                    torch.save(model, 'best_loss_model.pth') 

                if best_acc < correct_test/total_test:
                    best_acc = correct_test/total_test
                    torch.save(model, 'best_acc_model.pth') 

                print(f"\tIteration: {j}\t Loss: {running_loss/print_every:.4f} \t Val_Loss: {validation_loss / n_batches} \t Val_Acc: {correct_test/total_test}")
                running_loss = 0
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.show()
    #torch.save(model, 'model.pth')    

train(model, optimizer, criterion, n_epochs)



            

            



        
    

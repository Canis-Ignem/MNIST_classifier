from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class NNetwork(nn.Module):

    # Defining the layers, 128, 64, 16, 10 units each
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    # Forward function
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


model = NNetwork()
summary(model, (1,784))

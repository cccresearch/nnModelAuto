import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class Net(nn.Module):
    def __init__(self, shape, classes):
        super(Net, self).__init__()
        self.shape = shape # shape of input
        self.size = np.prod(shape) # size of input
        self.classes = classes # size of output
        self.accuracy = 0
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.size, 100),
            # torch.nn.ReLU(),
            torch.nn.Linear(100, self.classes)
        )

    def forward(self, x):
        x = x.view(-1, self.size) # reshape [1000, 1, 28, 28] into [1000, 28*28]
        x = self.model(x)
        return x

    def __str__(self):
        return str(self.model)


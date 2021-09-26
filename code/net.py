import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class Net(nn.Module):
    def __init__(self, shape, classes):
        super(Net, self).__init__()
        self.structure = {
            shape:shape,
            size:np.prod(shape),
            classes: classes,
            layers: []
        }
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.size, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, self.classes)
        )

    def forward(self, x):
        x = x.view(-1, self.size) # reshape [1000, 1, 28, 28] into [1000, 28*28]
        x = self.model(x)
        return x

    def __str__(self):
        return str(self.model)


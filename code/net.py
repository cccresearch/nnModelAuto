import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class Net(nn.Module):
    def __init__(self, shape, classes):
        super(Net, self).__init__()
        self.model = {
            shape:shape,
            size:np.prod(shape),
            classes: classes,
            layers:[]
        }
        self.buildNet()

    def self.buildNet():
        '''
        self.model.layers = [
            {"type": "flatten", inShape: [28, 28] }, # outshape 自動變為 [28*28]
            {"type": "linear", inShape: [28*28], outShape: [50] },
            {"type": "relu" }, # relu 大小自動使用前面的 outShape
            {"type": "linear", outshape:[10] } # 沒指定 inShape, 預設使用前面的 outShape 
        ]
        self.net = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear([28*28], 50),
            nn.ReLU(),
            nn.Linear(50, [10])
        )
        '''
        self.net = torch.nn.Sequential()
        for layer in self.model.layers:
            ltype = layer['type']
            nlayer = None               
            if ltype=='linear':
                nlayer = torch.nn.Linear(layer.inShape[0], layer.outShape[0])
            elif ltype=='relu':
                nlayer = torch.nn.ReLU()
            elif ltype=='sigmoid'
                nlayer = torch.nn.Sigmoid()
            else:
                raise Error('layer type unknown')

    def forward(self, x):
        x = x.view(-1, self.size) # reshape [1000, 1, 28, 28] into [1000, 28*28]
        x = self.model(x)
        return x

    def __str__(self):
        return str(self.model)


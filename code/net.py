import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import json

class Net(nn.Module):
    def __init__(self, inShape, outShape):
        super(Net, self).__init__()
        self.model = {
            "inShape":inShape,
            "outShape": outShape,
            "layers":[
                {"type": "flatten" }, # outshape 自動變為 [28*28]
                {"type": "linear", "inShape": [28*28], "outShape": [50] },
                {"type": "relu" }, # relu 大小自動使用前面的 outShape
                {"type": "linear", "outShape":[10] } # 沒指定 inShape, 預設使用前面的 outShape 
            ]
        }
        self.buildNet()

    def buildNet(self):
        print('model:', json.dumps(self.model, indent=2))

        self.net = torch.nn.Sequential()
        layers = self.model["layers"]
        shape = self.model["inShape"]
        for i in range(len(layers)):
            layer = layers[i]
            t = layers[i]['type']
            rlayer = None               
            if t=='flatten':
                rlayer = nn.Flatten()
            elif t=='linear':
                inSize = np.prod(shape).item()
                outSize = np.prod(layer['outShape']).item()
                rlayer = nn.Linear(inSize, outSize)
                shape = layer['outShape']
            elif t=='relu':
                rlayer = nn.ReLU()
            elif t=='sigmoid':
                rlayer = nn.Sigmoid()
            else:
                raise Exception(f'layer type {t} unknown')
            self.net.add_module(str(i), rlayer)
        
        print('net:', str(self.net))

    def forward(self, x):
        x = self.net(x)
        return x

    def __str__(self):
        return str(self.net)


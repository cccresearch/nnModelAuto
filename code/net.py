import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import json

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def build(self, model):
        self.model = model
        print('model:', json.dumps(self.model, indent=2))

        self.net = torch.nn.Sequential()
        layers = self.model["layers"]
        shape = self.model["in_shape"]
        for i in range(len(layers)):
            layer = layers[i]
            t = layers[i]['type']
            rlayer = None               
            if t=='flatten':
                rlayer = nn.Flatten()
            elif t=='linear':
                inSize = np.prod(shape).item()
                outSize = np.prod(layer['out_shape']).item()
                rlayer = nn.Linear(inSize, outSize)
                shape = layer['out_shape']
            elif t=='relu':
                rlayer = nn.ReLU()
            elif t=='sigmoid':
                rlayer = nn.Sigmoid()
            else:
                raise Exception(f'layer type <{t}> unknown')
            self.net.add_module(str(i), rlayer)
        
        print('net:', str(self.net))

    def forward(self, x):
        x = self.net(x)
        return x

    def __str__(self):
        return str(self.net)

    @staticmethod
    def base_model(in_shape, out_shape):
        net = Net()
        model = {
            "in_shape": in_shape,
            "out_shape": out_shape,
            "layers":[
                {"type": "flatten" },
                {"type": "linear", "out_shape": out_shape }
            ]
        }
        net.build(model)
        return net

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import json
import copy
from torchsummary import summary

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

	def build(self, model):
		self.model = copy.deepcopy(model)
		# print('model:', json.dumps(self.model, indent=2))
		self.net = torch.nn.Sequential()
		layers = self.model["layers"]
		shape = [1,1]+self.model["in_shape"] # MNIST 的 data shape 是 4 維 (batch_size, channels, 28, 28)
		x = torch.randn(shape)
		for i in range(len(layers)):
			layer = layers[i]
			t = layers[i]['type']
			rlayer = None               
			if t=='Flatten':
				rlayer = nn.Flatten()
			elif t=='Linear':
				in_features = shape[1]
				out_features = layer['out_features']
				rlayer = nn.Linear(in_features, out_features)
			elif t=='ReLU':
				rlayer = nn.ReLU()
			elif t=='Sigmoid':
				rlayer = nn.Sigmoid()
			elif t=='Conv2d':
				in_channels = shape[1]
				out_channels = layer['out_channels']
				kernel_size = layer.get('kernel_size', 3)
				stride = layer.get('stride', 1)
				rlayer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
			elif t=='AvgPool2d':
				kernel_size = layer['kernel_size']
				rlayer = nn.AvgPool2d(kernel_size)
			else:
				raise Exception(f'layer type <{t}> unknown')

			self.net.add_module(str(i), rlayer)
			print(i, ':rlayer=', rlayer)
			in_shape = x.size()
			#print('in_shape=', in_shape)
			x = rlayer(x)
			out_shape = x.size()
			#print('out_shape=', out_shape)
			shape = out_shape
		
		print('shape=', shape)
		self.net.add_module("out", nn.Linear(shape[1], self.model["out_shape"][0]))
		print('build:net=', self.net)
		self.model['parameter_count'] = self.parameter_count()

	def parameter_count(self):
		return sum(p.numel() for p in self.net.parameters())
		
	def forward(self, x):
		x = self.net(x)
		return x

	def __str__(self):
		return str(self.net)+f"\nparameter_count={self.parameter_count()}"

	@staticmethod
	def base_model(in_shape, out_shape):
		net = Net()
		model = {
			"in_shape": in_shape,
			"out_shape": out_shape,
			"layers":[
				{"type": "Flatten" }
			]
		}
		net.build(model)
		return net

	'''
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
	'''

	@staticmethod
	def cnn_model(in_shape, out_shape):
		net = Net()
		model = {
			"in_shape": in_shape,
			"out_shape": out_shape,
			"layers":[
				{"type": "Conv2d", "out_channels":6, "kernel_size":5, "stride": 1 },
				{"type": "AvgPool2d", "kernel_size":2 },
				{"type": "Conv2d", "out_channels":16, "kernel_size":5 },
				{"type": "AvgPool2d", "kernel_size":2 },
				{"type": "Flatten" },
				{"type": "Linear", "out_features":120 },
				{"type": "ReLU" },
				{"type": "Linear", "out_features":84 },
				{"type": "ReLU" },
			]
		}
		print("build cnn model:", model)
		net.build(model)
		return net
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import json
import copy
from torchsummary import summary

class ConvPool2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super(ConvPool2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
		self.pool = nn.AvgPool2d(kernel_size=2)

	def forward(self, x):
		x = self.conv(x)
		x = self.pool(x)
		return x

class LinearReLU(nn.Module):
	def __init__(self, in_features, out_features):
		super(LinearReLU, self).__init__()
		self.linear = nn.Linear(in_features, out_features)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.linear(x)
		x = self.relu(x)
		return x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

	def build(self, model):
		self.model = copy.deepcopy(model)
		# print('model:', json.dumps(self.model, indent=2))
		self.net = torch.nn.Sequential()
		layers = self.model["layers"]
		self.in_shapes = [[]]*len(layers)
		shape = [1,1]+self.model["in_shape"] # MNIST 的 data shape 是 4 維 (batch_size, channels, 28, 28)
		x = torch.randn(shape)
		for i in range(len(layers)):
			self.in_shapes[i] = shape
			layer = layers[i]
			t = layers[i]['type']
			rlayer = None               
			if t=='Flatten':
				rlayer = nn.Flatten()
			elif t in ['Linear', 'LinearReLU']:
				in_features = shape[1]
				out_features = layer['out_features']
				if t == 'Linear':
					rlayer = nn.Linear(in_features, out_features)
				else: # LinearReLU
					rlayer = LinearReLU(in_features, out_features)
			elif t=='ReLU':
				rlayer = nn.ReLU()
			elif t=='Sigmoid':
				rlayer = nn.Sigmoid()
			elif t in ['Conv2d', 'ConvPool2d']:
				in_channels = shape[1]
				out_channels = layer['out_channels']
				kernel_size = layer.get('kernel_size', 3)
				stride = layer.get('stride', 1)
				if t=="Conv2d":
					rlayer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
				else: # ConvPool2d
					rlayer = ConvPool2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
			elif t=='AvgPool2d':
				kernel_size = layer.get('kernel_size', 2)
				rlayer = nn.AvgPool2d(kernel_size)
			else:
				raise Exception(f'layer type <{t}> unknown')

			self.net.add_module(str(i), rlayer)
			# print(i, ':rlayer=', rlayer)
			in_shape = x.size()
			# print('in_shape=', in_shape)
			x = rlayer(x)
			out_shape = x.size()
			# print('out_shape=', out_shape)
			shape = out_shape
		
		# print('shape=', shape)
		self.net.add_module("out", nn.Linear(shape[1], self.model["out_shape"][0]))
		# print('build:net=', self.net)
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

	@staticmethod
	def cnn_model(in_shape, out_shape):
		net = Net()
		model = {
			"in_shape": in_shape,
			"out_shape": out_shape,
			"layers":[
				{"type": "ConvPool2d", "out_channels":6, "kernel_size":5, "stride": 1 },
				{"type": "Conv2d", "out_channels":16, "kernel_size":5 },
				{"type": "AvgPool2d", "kernel_size":2 },
				{"type": "Flatten" },
				{"type": "LinearReLU", "out_features":120 },
				{"type": "Linear", "out_features":84 },
				{"type": "ReLU" },
			]
		}
		print("build cnn model:", model)
		net.build(model)
		return net

	'''
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
	'''

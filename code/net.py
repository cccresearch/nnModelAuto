import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import json
import copy

def linearLayer(in_shape, out_shape):
	in_size = np.prod(in_shape).item()
	out_size = np.prod(out_shape).item()
	return nn.Linear(in_size, out_size)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

	def build(self, model):
		self.model = copy.deepcopy(model)
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
				rlayer = linearLayer(shape, layer['out_shape'])
				shape = layer['out_shape']
			elif t=='relu':
				rlayer = nn.ReLU()
			elif t=='sigmoid':
				rlayer = nn.Sigmoid()
			else:
				raise Exception(f'layer type <{t}> unknown')
			self.net.add_module(str(i), rlayer)
		
		self.net.add_module("out", linearLayer(shape, self.model["out_shape"]))
		self.model['parameter_count'] = self.parameter_count()
		print('net:', str(self.net))

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
				{"type": "flatten" }
			]
		}
		net.build(model)
		return net

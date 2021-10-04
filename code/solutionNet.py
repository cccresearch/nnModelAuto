from net import Net
from solution import Solution
import random
import trainer
import copy
import model
import sys

types = ["ReLU", "Linear", "Conv2d", "AvgPool2d"]
sizes = [ 8, 16, 32, 64, 128, 256 ] # 限縮大小選取範圍，避免太多鄰居，所以不是所有整數都可以
channels = [ 1, 2, 4, 8, 16, 32 ]

def randomLayer():
	type1 = random.choice(types)
	# print('randomLayer:type1=', type1)
	if type1 == "Linear":
		k = random.choice(sizes)
		return {"type":"Linear", "out_features":k}
	elif type1 == "Conv2d":
		out_channels = random.choice(channels)
		return {"type":"Conv2d", "out_channels": out_channels}
	elif type1 == "AvgPool2d":
		return {"type":"AvgPool2d"}
	else:
		return {"type":type1}

def compatable(in_shape, newLayerType):
	if newLayerType in ["ReLU"]:
		return True
	elif len(in_shape) == 4 and newLayerType in ["Conv2d", "AvgPool2d", "Flatten"]:
		return True
	elif len(in_shape) == 2 and newLayerType in ["Linear"]:
		return True
	return False

class SolutionNet(Solution):
	def __init__(self, net):
		super(type(self), self).__init__(net)
		self.net = net

	def neighbor(self):
		model = copy.deepcopy(self.net.model)
		layers = model["layers"]
		in_shapes = self.net.in_shapes
		ops = ["insert", "update", "update", "delete"]
		success = False
		while not success:
			i = random.randint(0, len(layers)-1)
			layer = layers[i]
			op = random.choice(ops)
			newLayer = randomLayer()
			if not compatable(in_shapes[i], newLayer["type"]):
				continue
			if op == "insert":
				layers.insert(i, newLayer)
			elif op == "update":
				if layers[i]["type"] == "Flatten":
					continue
				else:
					layers[i] = newLayer
			break

		nNet = Net()
		nNet.build(model)
		return SolutionNet(nNet)
 
	def height(self):
		net = self.net
		if not model.exist(net):
			trainer.run(net)
		else:
			jsonObj = model.load(net)
			net.model['accuracy'] = jsonObj['model']['accuracy']
		
		return net.model['accuracy']-(net.model['parameter_count']/1000000)

	def __str__(self):
		return "{} height={:f}".format(self.net.model, self.height())


'''
	def neighbor(self):
		model = copy.deepcopy(self.net.model)
		layers = model["layers"]
		in_shapes = self.net.in_shapes
		ops = ["insert", "update", "update", "delete"]
		success = False
		while not success:
			success = True
			i = random.randint(0, len(layers)-1)
			layer = layers[i]
			op = random.choice(ops)
			if op == "insert":
				newLayer = randomLayer()
			#	if (compatable(in_shapes[i+1], newLayer))
			#		layers.insert(i+1, newLayer) # 插在第 i 層後面
			#	else:
					success = False
			# elif layer["type"]=="Flatten": # Flatten 層必須保留(只能有一個 Flatten 層)
			#	success = False
			elif op == "delete":
				del layers[i]
			elif op == "update":
				if (compatable(in_shapes[i], newLayer))
					layers[i] = randomLayer()
				else:
					success = False
			else:
				success = False

		nNet = Net()
		nNet.build(model)
		return SolutionNet(nNet)
'''

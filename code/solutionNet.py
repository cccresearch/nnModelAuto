from net import Net
from solution import Solution
import random
import trainer
import copy
import model
import sys

types = ["relu", "linear"] # "sigmoid",...
sizes = [ 8, 16, 32, 64, 128, 256 ] # 限縮大小選取範圍，避免太多鄰居，所以不是所有整數都可以

def randomLayer():
	type1 = random.choice(types)
	print('randomLayer:type1=', type1)
	if type1 == "linear":
		k = random.choice(sizes)
		return {"type":"linear", "out_shape":[k]}
	else:
		return {"type":type1}

class SolutionNet(Solution):
	def __init__(self, net):
		super(type(self), self).__init__(net)
		self.net = net

	def neighbor(self):
		model = copy.deepcopy(self.net.model)
		layers = model["layers"]
		ops = ["insert", "update", "update", "delete"]
		success = False
		while not success:
			success = True
			i = random.randint(0, len(layers)-1)
			layer = layers[i]
			op = random.choice(ops)
			if op == "insert":
				layers.insert(i+1, randomLayer()) # 插在第 i 層後面
			elif layer["type"]=="flatten": # Flatten 層必須保留(只能有一個 Flatten 層)
				success = False
			elif op == "delete":
				del layers[i]
			elif op == "update":
				layers[i] = randomLayer()
			else:
				success = False

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

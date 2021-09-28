from net import Net
from solution import Solution
import random
import trainer
import copy
import model
import sys

types = ["relu", "linear", "conv2d", "avg_pool2d"]
sizes = [ 8, 16, 32, 64, 128, 256 ] # 限縮大小選取範圍，避免太多鄰居，所以不是所有整數都可以
channels = [ 1, 2, 4, 8, 16, 32 ]
'''
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
'''

def randomLayer():
	type1 = random.choice(types)
	# print('randomLayer:type1=', type1)
	if type1 == "linear":
		k = random.choice(sizes)
		return {"type":"linear", "out_features":k}
	elif type1 == "conv2d":
		out_channels = random.choice(channels)
		return {"type":"conv2d", "out_channels": out_channels}
	elif type1 == "avg_pool2d":
		return {"type":"avg_pool2d"}
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

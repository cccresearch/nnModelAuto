from net import Net
from solution import Solution
import random
import trainer
import copy
import model
import sys

types = ["relu", "linear"] # "sigmoid",...
sizes = [ 8, 16, 32, 64, 128, 256 ] # 限縮選取範圍，避免太多鄰居

def randomLayer():
	type1 = random.choice(types)
	if type1 == "linear":
		k = random.choice(sizes)
		return {"type":"linear", "out_shape":[k]}
	else:
		return {"type":type1}

class SolutionNet(Solution):
	def __init__(self, net):
		super(type(self), self).__init__(net)
		self.net = net

	'''
	3. 鄰居的產生方法
	* 選定一層，然後用《分裂、修改、新增、刪除》等方法進行鄰居產生
	* 全連接層適合用《分裂》產生鄰居。(例如 MNIST 562=>10, 分裂成 562=>50=>10)
	* RELU/Sigmoid 這類的層，可以新增與刪除。
	* 對於有參數的層，可以使用修改參數的方式產生鄰居。
	'''
	def neighbor(self):
		model = copy.deepcopy(self.net.model)
		layers = model["layers"]
		ops = ["insert", "update", "update", "delete"]
		i = random.randint(0, len(layers)-1)
		op = random.choice(ops)

		if op == "delete":
			del layers[i]
		elif op == "update":
			layers[i] = randomLayer()
		else: # op == "insert":
			layers.insert(i, randomLayer())
		
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
		print('net=', net)
		h = net.model['accuracy']-(net.model['parameter_count']/10000)
		net.model['height'] = h
		print('model=', net.model)
		return h

	def str(self):
		return "height({})={:f}".format(self.net.model, self.height())


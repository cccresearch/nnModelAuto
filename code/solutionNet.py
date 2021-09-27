from net import Net
from solution import Solution
import random
import trainer
import copy
import model
import sys

actTypes = ["relu", "sigmoid" ]

class SolutionNet(Solution):
	def __init__(self, net):
		super(SolutionNet, self).__init__(net)
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
		success = False
		while not success:
			i = random.randint(0, len(layers)-1)
			t = layers[i]['type']
			if t in actTypes:
				layers[i]["type"] = random.choice(actTypes)
				success = True
			elif t=="linear":
				out_shape = layers[i]["out_shape"]
				layers.insert(i, {"type":"relu"})
				k = random.randint(2, 8) # 前一個 linear 層 out_shape 增加 2 到 8 倍
				layers.insert(i, {"type":"linear", "out_shape":[x * k for x in out_shape]})
				success = True
		
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
		return net.model['accuracy']

	def str(self):
		return "height({})={:f}".format(self.net.model, self.height())


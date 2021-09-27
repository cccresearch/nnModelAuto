from net import Net
from solution import Solution
from random import random, randint
import trainer
import copy
import model
import sys

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
	def neighbor(self): # 鄰居函數
		model = copy.deepcopy(self.net.model)
		layers = model["layers"]
		layers.append({"type": "relu"})
		nNet = Net()
		nNet.build(model)
		return SolutionNet(nNet)

	def height(self): #  能量函數
		net = self.net
		if not model.exist(net):
			trainer.run(net)
		else:
			jsonObj = model.load(net)
			net.model['accuracy'] = jsonObj['model']['accuracy']
		
		print('net=', net)
		return net.model['accuracy']

	def str(self):    #  將解答轉為字串的函數，以供列印用。
		return "height({})={:f}".format(self.net.model, self.energy())


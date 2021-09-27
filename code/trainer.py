import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from random import random
import hashlib
import json
import os

# n_epochs = 3
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('files/', train=True, download=True,
							 transform=torchvision.transforms.Compose([
								 torchvision.transforms.ToTensor(),
								 torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
	batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('files/', train=False, download=True,
							 transform=torchvision.transforms.Compose([
								 torchvision.transforms.ToTensor(),
								 torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
	batch_size=batch_size_test, shuffle=True)

'''
train_losses = None
train_counter = None
test_losses = None
test_counter = None
network = []
optimizer = []
'''

def init(net):
	global train_losses, train_counter, test_losses, test_counter, network, optimizer
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
	network = net
	optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

def train(epoch):
	network.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if batch_idx > 300: break
		optimizer.zero_grad()
		output = network(data)
		loss = F.nll_loss(F.log_softmax(output, dim=1), target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append(
			(batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
		
			# torch.save(network.state_dict(), 'results/model.pth')
			# torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test():
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = network(data)
			test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)
		accuracy = 100. * correct / len(test_loader.dataset)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset), accuracy))
		network.model['accuracy'] = accuracy.item()

def modelHash(net):
	sha = hashlib.sha256()
	sha.update(str(net).encode())
	return sha.hexdigest()

def modelExist(net):
	filename = modelHash(net)
	return os.path.isfile(f'model/{filename}.json')

def modelLoad(net):
	filename = modelHash(net)
	jsonFile = open(f'model/{filename}.json', "rt")
	jsonStr = jsonFile.read()
	jsonObj = json.loads(jsonStr)
	jsonFile.close()
	torch.load(net.state_dict(), f'model/{filename}.pt')

def modelSave(net):
	filename = modelHash(net)
	jsonObj = {
		"filename": filename,
		"model":net.model
	}
	jsonFile = open(f'model/{filename}.json', "wt")
	jsonFile.write(json.dumps(jsonObj, indent=2))
	jsonFile.close()
	torch.save(net.state_dict(), f'model/{filename}.pt')

def run(net):
	init(net)
	test()
	for epoch in range(1, n_epochs + 1):
		train(epoch)
		test()
	modelSave(net)

module = __import__(sys.argv[1])
Net = module.Net
# net = Net([28, 28], 10)
net = Net([28, 28], [10])
if modelExist(net):
	print('model exist!')
	print('net.model[0]=', net.model[0])
	print('net.model[0].in_features=', net.model[0].in_features)
	print('net.model[0].out_features=', net.model[0].out_features)
else:
	run(net)

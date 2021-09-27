import torch
import hashlib
import json
import os

def hash(net):
	sha = hashlib.sha256()
	sha.update(str(net).encode())
	return sha.hexdigest()

def exist(net):
	filename = hash(net)
	return os.path.isfile(f'model/{filename}.json')

def load(net):
	filename = hash(net)
	jsonFile = open(f'model/{filename}.json', "rt")
	jsonStr = jsonFile.read()
	jsonObj = json.loads(jsonStr)
	jsonFile.close()
	# torch.load(net.state_dict(), f'model/{filename}.pt')
	return jsonObj

def save(net):
	filename = hash(net)
	jsonObj = {
		"filename": filename,
		"model":net.model
	}
	jsonFile = open(f'model/{filename}.json', "wt")
	jsonFile.write(json.dumps(jsonObj, indent=2))
	jsonFile.close()
	torch.save(net.state_dict(), f'model/{filename}.pt')

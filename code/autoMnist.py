from hillClimbing import hillClimbing
from net import Net
import model
from solutionNet import SolutionNet

def start():
	net = Net.base_model([28,28], [10])
	if not model.exist(net):
		run(net)
	return SolutionNet(net)

hillClimbing(start(), 2, 2)

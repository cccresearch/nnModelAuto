from solution import Solution
from random import random, randint

class SolutionNet(Solution):
    def __init__(self, net):
        self.net = net
        
    def neighbor(self): # 鄰居函數


    def energy(self): #  能量函數
        return self.loss

    def str(self):    #  將解答轉為字串的函數，以供列印用。
        return "loss({:s})={:f}".format(str(self.v), self.energy())



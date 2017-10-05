from random import random
import math

class Node:
    # inputs: whether or not it's an output node, and activations of previous layer
    def __init__(self, isOutput, prevActivs):
        # instance variables
        self.isOutput = isOutput
        self.prevActivs = prevActivs
        self.weights = []
        self.activ = -1.0
        self.initWeights()
    
    # get activation of this node
    def getActiv(self):
        # if not already done, sum previous activations times weights, and pass into activation function
        if self.activ == -1.0:
            weightedSum = sum([prev * weight for prev in self.prevActivs for weight in self.weights])
            self.activ = self.activFunct(weightedSum)
        return self.activ
    
    # initialize random weights
    def initWeights(self):
        for i in range(len(self.prevActivs)):
            self.weights[i] = random.randrange(-.1,.1)
    
    # node's activation function
    def activFunct(self, weightedSum):
        return 1 / (1 + math.pow(math.e, weightedSum))
    
    def sigma(self):
        # implement sigma function, or make subclass with different activation function?
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
        # by default, node superclass does not have an activation function
        return weightedSum

# Backpropagation node subclass
class BPNode(Node):
    
    def __init__(self, isOutput, prevActivs):
        # call constructor of super
        Node.__init__(self, isOutput, prevActivs)
        
    # use logistic activation function for backprop
    def activFunct(self, weightedSum):
        return 1 / (1 + math.pow(math.e, weightedSum))
    
    
# RBF node subclass
class RBFNode(Node):
    
    def __init__(self, isOutput, prevActivs, center, variance):
        # call constructor of super
        Node.__init__(self, isOutput, prevActivs)
        # assign center and variance to node
        self.center = center
        self.variance = variance
        
        

        phiValue = 0
        # TODO: Implement phi for RBF hidden nodes
        return phiValue
        
        
        
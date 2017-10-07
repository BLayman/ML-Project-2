import random
import math

# Node superclass
class Node:
    # inputs: whether or not it's an output node, and activations of previous layer
    def __init__(self, isOutput, weightNum):
        # instance variables
        self.weightNum = weightNum # number of input weights
        self.isOutput = isOutput # may be nice for simplifying ForwardProp
        self.weights = [] # input edge weights
        self.activ = -1.0 # default -1 value to show if activation has been set yet
        # initialize random weights
        self.initWeights()
        
    
    # get activation of this node
    def setActiv(self, prevActivs):
        # if not already done, sum previous activations times weights, and pass into activation function
        if self.activ == -1.0:
            weightedSum = sum([prev * weight for prev in prevActivs for weight in self.weights])
            print("weighted sum: " + str(weightedSum))
            self.activ = self.activFunct(weightedSum)
    
    #get activation of this node        
    def getActiv(self):
        return self.activ
    
    # initialize random weights
    def initWeights(self):
        for i in range(self.weightNum):
            randomNum = random.uniform(-.1,.1)
            self.weights.append(randomNum)
            print("weight: " + str(randomNum))
    
    # node's activation function
    def activFunct(self, weightedSum):
        # by default, Node superclass does not have an activation function
        return weightedSum

# Backpropagation node subclass
class BPNode(Node):
    
    def __init__(self, isOutput, weightNum):
        # call constructor of super
        Node.__init__(self, isOutput, weightNum)
        
    # use logistic activation function for backprop
    def activFunct(self, weightedSum):
        return 1 / (1 + math.pow(math.e, -1 * weightedSum))
    
    
# RBF node subclass
class RBFNode(Node):
    
    def __init__(self, isOutput, weightNum, center, variance):
        # call constructor of super
        Node.__init__(self, isOutput, weightNum)
        # assign center and variance to node
        self.center = center
        self.variance = variance
        
        
    def activFunct(self, weightedSum):
        phiValue = 0
        # TODO: Implement phi for RBF hidden nodes
        return phiValue
        
        
        
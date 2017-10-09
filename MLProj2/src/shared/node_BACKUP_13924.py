from random import random
import math

# Node superclass
class Node:
    # inputs: whether or not it's an output node, and activations of previous layer
    def __init__(self, isOutput):
        # instance variables
        self.isOutput = isOutput # may be nice for simplifying ForwardProp
        self.weights = [] # input edge weights
<<<<<<< HEAD
        self.partials  = [] # partial derivatives of weights with respect to error
        self.delta = -1.0 # default -1 value to show that delta has been set yet
        self.activ = -1.0 # default -1 value to show that activation has been set yet
        # initialize random weights
=======
        self.activ = -1.0 # default -1 value to show if activation has been set yet
         # initialize random weights
>>>>>>> 367ce9debb3fd43b576c0b6fdc10a90c325d181c
        self.initWeights()
        
    
    #get activation of this node        
    def getActiv(self):
        return self.activ
    
    # get activation of this node
    def setActiv(self, prevActivs):
        # if not already done, sum previous activations times weights, and pass into activation function
        if self.activ == -1.0:
            weightedSum = sum([prev * weight for prev in prevActivs for weight in self.weights])
            self.activ = self.activFunct(weightedSum)
    
    def setPartials(self, partials):
        self.partials = partials
        
    # updates weights using partial derivatives and learning rate alpha
    def updateWeights(self, alpha):
        for i in range(self.weights):
            self.weights[i] -= alpha * self.partials[i]
        
    # initialize random weights
    def initWeights(self):
        for i in range(len(self.prevActivs)):
            self.weights[i] = random.randrange(-.1,.1)
    
    # node's activation function
    def activFunct(self, weightedSum):
        # by default, Node superclass does not have an activation function
        return weightedSum

# Backpropagation node subclass
class BPNode(Node):
    
    def __init__(self, isOutput):
        # call constructor of super
        Node.__init__(self, isOutput)
        
    # use logistic activation function for backprop
    def activFunct(self, weightedSum):
        return 1 / (1 + math.pow(math.e, weightedSum))
    
    
# RBF node subclass
class RBFNode(Node):
    output = 0
    def __init__(self, isOutput):
        # call constructor of super
        Node.__init__(self, isOutput)

        
        
    def activFunct(self, inputVal):
        weightedSum = sum([inp * weight for inp in inputVal for weight in self.weights])
        self.output = weightedSum
        return weightedSum
        
        
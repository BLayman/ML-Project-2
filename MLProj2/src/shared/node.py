import random
import math
from operator import add

# Node superclass
class Node:
    # inputs: whether or not it's an output node, and activations of previous layer
    def __init__(self, weightNum):
        # instance variables
        self.weightNum = weightNum # number of input weights
        self.weights = [] # input edge weights
        self.avgPartials = [] # partial derivatives of weights with respect to error
        self.partialsSum = []
        self.delta = -1.0 # default -1 value to show that delta has been set yet
        self.activ = -1.0 # default -1 value to show that activation has been set yet
        # initialize random weights
        self.initWeights()
        
    def setDelta(self, delta):
        self.delta = delta
        
    def getDelta(self):
        return self.delta
    
    def getWeights(self):
        return self.weights
    
    # get activation of this node        
    def getActiv(self):
        return self.activ
    
    def getPartials(self):
        return self.avgPartials
    
    # manually set activation for first layer
    def setActiv(self, activ):
        self.activ = activ
    
    # calculate and set activation of this node
    def calcActiv(self, prevActivs):
        # if not already done, sum previous activations times weights, and pass into activation function
        if self.activ == -1.0:
            weightedSum = sum([prev * weight for prev in prevActivs for weight in self.weights])
            # print("weighted sum: " + str(weightedSum))
            self.activ = self.activFunct(weightedSum)
            
    # repeatedly called by BackProp class
    def addPartials(self, partials):
        print("partials array: " + str(partials))
        if(self.partialsSum == []):
            self.partialsSum = partials
        else:
            self.partialsSum = map(add, self.partialsSum, partials)
    
    # called by GradientDescent class
    # updates weights using partial derivatives and learning rate alpha
    def updateWeights(self, alpha, dataSetSize):
        # average out partial derivative from sum
        self.avgPartials = [pSum / dataSetSize for pSum in self.partialsSum]
        for i in range(len(self.weights)):
            self.weights[i] -= alpha * self.avgPartials[i]
        
    # initialize random weights
    def initWeights(self):
        for i in range(self.weightNum):
            randomNum = random.uniform(-.1,.1)
            self.weights.append(randomNum)
            # print("weight: " + str(randomNum))
    
    # node's activation function
    def activFunct(self, weightedSum):
        # by default, Node superclass does not have an activation function
        return weightedSum

# Backpropagation node subclass
class BPNode(Node):
    
    def __init__(self, weightNum):
        # call constructor of super
        Node.__init__(self, weightNum)
        
    # use logistic activation function for backprop
    def activFunct(self, weightedSum):
        return 1 / (1 + math.pow(math.e, -1 * weightedSum))
    
    
# RBF node subclass
class RBFNode(Node):
    
    def __init__(self, weightNum):
        # call constructor of super
        Node.__init__(self, weightNum)
        self.partialsSum = [weightNum]
        
        # assign center and variance to node
        
        
    def activFunct(self, node):
        output = 0
        partials = []
        for i in range(len(node.phiValues)):
            #Calculates the output from a  given input
            output += node.phiValues[i] * self.weights[i]
            node.output = output
            #Adds the derivitive with respect to the weight to partialSum
        for j in range(len(node.phiValues)):
            partials[i] = (node.expectedOutput - output) * node.phiValue[i]
        self.addPartials(partials)

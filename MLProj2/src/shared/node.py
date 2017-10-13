import random
import math
from operator import add

# Node superclass
class Node:
    # inputs: whether or not it's an output node, and activations of previous layer
    def __init__(self, weightNum):
        # instance variables
        self.errorDiff = 0.0
        self.weightNum = weightNum # number of input weights
        self.weights = [] # input edge weights
        self.avgPartials = [] # partial derivatives of weights with respect to error
        self.prevWeightChanges = []
        self.partialsSum = []
        self.delta = -1.0 # default -1 value to show that delta has been set yet
        self.activ = 1.0 # default activation
        # initialize random weights
        self.initWeights() 
        #self.initTestWeights()
        
    def setDelta(self, delta):
        self.delta = delta
        
    def setErrorDiff(self, errorDiff):
        self.errorDiff = errorDiff
        
    def getErrorDiff(self):
        return self.errorDiff
    
    def getDelta(self):
        return self.delta
    
    def setPartialsSum(self, pSum):
        self.partialsSum = pSum
    
    def getWeights(self):
        return self.weights
    
    # get activation of this node        
    def getActiv(self):
        return self.activ
    
    def getPartials(self):
        return self.avgPartials
    
    def getParialsSum(self):
        return self.partialsSum
    
    # manually set activation for first layer
    def setActiv(self, activ):
        self.activ = activ
    
    # calculate and set activation of this node
    def calcActiv(self, prevActivs):
        # sum previous activations times weights, and pass into activation function
        weightedSum = 0
        for i in range(len(self.weights)):
            weightedSum += self.weights[i] * prevActivs[i]
        self.activ = self.activFunct(weightedSum)
            
    # repeatedly called by BackProp class
    def addPartials(self, partials):
        #print("partials array: " + str(partials))
        if(self.partialsSum == []):
            self.partialsSum = partials
        else:
            for i in range(len(self.partialsSum)):
                self.partialsSum[i] += partials[i]
    
    # called by GradientDescent class
    # updates weights using partial derivatives and learning rate alpha
    def updateWeights(self, alpha, dataSetSize, regParam):
        momentumParam = 0
        currWeightChanges = []
        # average out partial derivative from sum
        self.avgPartials = [pSum / dataSetSize for pSum in self.partialsSum]
        for i in range(len(self.weights)):
            # if not bias term, use regularization
            if (i != len(self.weights)-1):
                weightChange = -alpha * ((self.avgPartials[i] + (regParam/dataSetSize) * self.weights[i]))
                self.weights[i] += weightChange
                currWeightChanges.append(weightChange)
            # if bias term, don't use regularization
            else:
                weightChange = -alpha * self.avgPartials[i]
                self.weights[i] += weightChange
                currWeightChanges.append(weightChange)
            # add momentum
            if(len(self.prevWeightChanges) > 0):
                #print("add momentum: " + str(self.prevWeightChanges[i]))
                self.weights[i] += momentumParam * (self.prevWeightChanges[i])      
        self.prevWeightChanges = currWeightChanges
                
        
    # initialize random weights
    def initWeights(self):
        for i in range(self.weightNum):
            randomNum = random.uniform(-.5,.5)
            self.weights.append(randomNum)
    
    def initTestWeights(self):
        for i in range(self.weightNum):
            self.weights.append(-.1)
            
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
    
class BiasNode(Node):
    
    def __init__(self, weightNum):
        # call constructor of super
        Node.__init__(self, weightNum)
        
    # activation stays the same in bias nodes
    def calcActiv(self, prevActivs):
        self.activ = self.activ
    
    
# RBF node subclass
class RBFNode(Node):
    maxError = 0
    def __init__(self, weightNum):
        # call constructor of super
        Node.__init__(self, weightNum)
        self.weightNum = weightNum
        self.weights.append(random.uniform(-.1,.1))
        self.partialsSum = [0] * (weightNum + 1)
        
        # assign center and variance to node
        
        
    def activeFunct(self, RbNode, index):
        output = 0
        
        print(RbNode.phiValues)
        for i in range(len(RbNode.phiValues)):
            #Calculates the output from a  given input
            output += RbNode.phiValues[i] * self.weights[i]
        if(len(RbNode.output) == index):
            RbNode.output.append(output)
        else:
            RbNode.output[index] = output
            #Adds the derivitive with respect to the weight to partialSum
        for j in range(len(RbNode.phiValues)):
            error = (RbNode.expectedOut - output)
            if error > self.maxError:
                self.maxError = error
            temp = self.partialsSum[j]
            temp += error * RbNode.phiValues[j]
            self.partialsSum[j] = temp
        
    def updateWeights(self, alpha, dataSetSize):   
        stop = True
        for k in range(len(self.partialsSum)):
            self.partialsSum[k] /= float(dataSetSize)
        self.avgPartials = self.partialsSum
        print(self.avgPartials)
        for i in range(len(self.weights)):
            self.weights[i] += alpha * self.avgPartials[i]
            if (abs(self.avgPartials[i]) > 0.01):
                stop = False
        print(self.weights, "weights")
        return stop
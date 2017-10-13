'''
Created on Oct 12, 2017

@author: Carsen
'''
from shared import node
from radialBasis import RbNode
from radialBasis import RbNodeHidden
from shared.gradientDescent import GradientDescent
from shared.node import Node


class radialBasisOut:
    dataPoints = [[]]
    expectedOutput = []
    k = 0
    numOut = 0
    means = [[]]
    inputNodes = []
    hiddenNodes = []
    outputNodes = []
    def __init__(self, dataPoints, expectedOutput,k,numOut,means):
        self.dataPoints = dataPoints
        self.expectedOutput = expectedOutput
        self.k = k
        self.numOut = numOut
        self.means = means
        self.inputNodes = [RbNode] * len(dataPoints)
        self.createInputNodes()
        self.createHiddenNodes()
        self.createOutNodes()
        self.calcPhiVals()
        self.calcOutValues()
        self.train(1)
        print(self.outputNodes[0].weights, "final weights")
        print()
        print(self.test([-.1,-.1], .009))
        print(self.test([-.1,-.5], -0.11499999999999999))
        print(self.test([.7,.2], .49799999999999994))
        
    #kmeans will return a means[] list, with coresponding tuples, with sigma
    
    def createInputNodes(self):
        for i in range(len(self.dataPoints)):
            #temp1 = self.dataPoints[i]
            #expected = self.expectedOutput[i]
            #temp = RbNode.RbNode(temp1, expected, self.k)
            self.inputNodes[i] = RbNode.RbNode(self.dataPoints[i], self.expectedOutput[i], self.k)
    def createHiddenNodes(self):
        print(len(self.means))
        for i in range(len(self.means)):
            self.hiddenNodes.append(RbNodeHidden.RbNodeHidden(self.means[i][0], self.means[i][1]))
        print(len(self.hiddenNodes))
    def createOutNodes(self):
        for i in range(self.numOut):
            self.outputNodes.append(node.RBFNode(self.k))
    def calcPhiVals(self):
        for i in range(len(self.inputNodes)):
            for j in range(len(self.hiddenNodes)):
                self.inputNodes[i].addPhi(self.hiddenNodes[j], j)
            self.inputNodes[i].phiValues.append(1)
            
    def calcOutValues(self):
        for i in range(len(self.inputNodes)):
            for j in range(len(self.outputNodes)):
                self.outputNodes[j].activeFunct(self.inputNodes[i] , j)
    def train(self, alpha):
        network = [[RbNode] * self.numOut] * 1
        for i in range(len(self.outputNodes)):
            network[0][i] = self.outputNodes[i]
        #descent1 = GradientDescent(network, alpha, len(self.inputNodes))
        
        stop = False
        count = 0
        while stop != True:
            stop = True
            for i in range(len(self.outputNodes)):
                self.outputNodes[i].averagePartials = []
                self.outputNodes[i].partialsSum = [0] * (self.k + 1)
            self.calcOutValues()
            for i in range(len(self.outputNodes)):
                stop = self.outputNodes[i].updateWeights(1,len(self.inputNodes))
                #if (self.outputNodes[i].maxError > 0.001):
                #   print(self.outputNodes[i].maxError, "maxE")
                 #   stop = False
            self.calcOutValues()
            count += 1
        print("rate", count)
    def test(self, inputVector, expectedOut):
        testPhi = []
        node = RbNode.RbNode(inputVector, expectedOut, self.k)
        for i in range(len(self.hiddenNodes)):
            testPhi.append(self.hiddenNodes[i].calcPhi(node.inputVector))
        testPhi.append(1)
        node.phiValues = testPhi
        for j in range(len(self.outputNodes)):
            self.outputNodes[j].activeFunct(node, j)
        return node.output
    
        
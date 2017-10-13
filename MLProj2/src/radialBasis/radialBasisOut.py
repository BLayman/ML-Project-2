'''
Created on Oct 12, 2017

@author: Carsen
'''
from shared import node
from radialBasis import KMeans
from radialBasis import RbNode
from radialBasis import RbNodeHidden
from shared.gradientDescent import GradientDescent
from shared.node import Node
import math


class radialBasisOut:
    dataPoints = [[]]
    expectedOutput = []
    k = 0
    numOut = 0
    means = [[]]
    inputNodes = []
    hiddenNodes = []
    outputNodes = []
    alpha = 1
    #accepts a [[]] of data points, a [] of expected outputs corresponding to each data point, integer for k means, integer for number of output nodes
    #a [[]] of test points, and a [] of outputs for the test points. 
    #Returns a [[]] of errors with the inner array[i] corresponding to the errors of each test point, for each i'th output node 
    def __init__(self, dataPoints, expectedOutput,k,numOut, testPoints, testExpectedOut):
        self.dataPoints = dataPoints
        self.expectedOutput = expectedOutput
        self.k = k
        self.numOut = numOut
        means1 = KMeans.KMeans(dataPoints, k)
        means1.calcMeans()
        means1.reCluster()
        self.means = means1.calcSigma()
        self.inputNodes = [RbNode] * len(dataPoints)
        self.createInputNodes()
        self.createHiddenNodes()
        self.createOutNodes()
        self.calcPhiVals()
        self.calcOutValues()
        self.train(self.alpha)
        return self.test(testPoints, testExpectedOut)        
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
    def test(self, inputVectors, expectedOut):
        nodes = []
        testErrors = [[0] for i in range(len(inputVectors))] * range(self.numOut)
        for i in(len(inputVectors)):
            testPhi = []
            node = RbNode.RbNode(inputVectors, expectedOut, self.k)
            for i in range(len(self.hiddenNodes)):
                testPhi.append(self.hiddenNodes[i].calcPhi(node.inputVector))
            testPhi.append(1)
            node.phiValues = testPhi
            nodes.append(node)

        for j in range(len(self.outputNodes)):
            for i in range(len(nodes)):
                out = self.outputNodes[j].activeFunct(nodes[i], j)
                nodes[i].outputs[j] = out
                testErrors[j][i] = .5 * math.pow((expectedOut[i] - out), 2)
        return testErrors
    
        
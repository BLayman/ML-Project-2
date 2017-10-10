from shared import node
from radialBasis import RbNode
from radialBasis import RbNodeHidden
from shared.gradientDescent import GradientDescent
from shared.node import Node


class RadialBasis:
    dataPoints = [[]]
    expectedOutput = []
    k = 0
    numOut = 0
    means = [[]]
    inputNodes = []
    hiddenNodes = []
    outputNodes = []
    def __init__(self, dataPoints, expectedOutput,k,numOut):
        self.dataPoints = dataPoints
        self.expectedOutput = expectedOutput
        self.k = k
        self.numOut = numOut
        self.createInputNodes()
        self.createHiddenNodes()
        self.createOutNodes()
        self.calcPhiVals()
        self.calcOutValues()
        
    #kmeans will return a means[] list, with coresponding tuples, with sigma
    
    def createInputNodes(self):
        for i in self.dataPoints:
            self.inputNodes.append(RbNode(self.dataPoints[i], self.expectedOutput[i]))
    def createHiddenNodes(self):
        for i in self.means:
            self.hiddenNodes.append(RbNodeHidden(self.means[i][0], self.means[i][1]))
    def createOutNodes(self):
        for i in range(self.numOut):
            self.outputNodes.append(node.RBFNode(self.k))
    def calcPhiVals(self):
        for i in range(len(self.inputNodes)):
            for j in range(len(self.hiddenNodes)):
                self.inputNodes[i].addPhi(self.hiddenNode[j])
    def calcOutValues(self):
        for i in range(len(self.inputNodes)):
            for j in range(len(self.outputNodes)):
                self.outputNodes[j].activeFunct(self.inputNodes[i])
    def train(self, alpha):
        network = [[]]
        for i in range(len(self.outputNodes)):
            network.append(self.outputNodes[i])
        descent1 = GradientDescent(network, alpha, len(self.inputNodes))
        stop = False
        while stop != True:
            stop = descent1.updateWeights()
            for i in range(len(self.outputNodes)):
                self.outputNodes[i].averagePartials = []
                self.outputNodes[i].partialsSum = []
            self.calcOutValues()
    
    def test(self, inputVector, expectedOut):
        testPhi = []
        node = RbNode(inputVector, expectedOut)
        for i in range(len(self.hiddenNodes)):
            self.testPhi.append(self.hiddenNodes[i].calcPhi(node.inputVector))
        for j in self.outputNodes:
            self.outputNodes[j].acvtiveFunct(node)
        return node.output 
    
        
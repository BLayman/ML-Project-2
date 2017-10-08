from shared import node
from radialBasis import RbNode
from radialBasis import RbNodeHidden


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
            self.outputNodes.append(node(True))
    def calcPhiVals(self):
        for i in self.inputNodes:
            for j in self.hiddenNodes:
                self.inputNodes[i].addPhi(self.hiddenNode(j))
    def calcOutValues(self):
        for i in self.inputNodes:
            for j in self.outputNodes:
                temp = self.outputNodes[j].activFunct(self.inputNodes[i].phiValues)
                self.inputNodes[i].output[j] = temp
    def train(self):
        pass
    
    def test(self, inputVector):
        testPhi = []
        testOut = []
        for i in self.hiddenNodes:
            self.testPhi.append(self.hiddenNodes[i].calcPhi(inputVector))
        for j in self.outputNodes:
            self.testOut.append(self.outputNodes.activFunct(self.tempPhi))
        return testOut    
    
        
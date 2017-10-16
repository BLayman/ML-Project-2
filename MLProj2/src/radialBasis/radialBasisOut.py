'''
Created on Oct 12, 2017

@author: Carsen
'''
from shared import node
from radialBasis import KMeans
from radialBasis import RbNode
from radialBasis import RbNodeHidden
from experiment import generate_data
import math
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pylab as pl



class radialBasisOut:
    dataPoints = [[]]
    expectedOutput = []
    k = 0
    numOut = 0
    means = [[]]
    inputNodes = []
    hiddenNodes = []
    outputNodes = []
    errors =[]
    alpha = 0
    #accepts a [[]] of data points, a [] of expected outputs corresponding to each data point, integer for k means, integer for number of output nodes
    def __init__(self, dataPoints, expectedOutput,k,numOut,alpha):
        self.dataPoints = dataPoints
        self.expectedOutput = expectedOutput
        self.k = k
        self.numOut = numOut
        self.alpha = alpha
               
    #upon this call, there will be a trained network that will be able to be tested by the function test()
    def createNetwork(self):
        means1 = KMeans.KMeans(self.dataPoints, self.k)
        means1.calcMeans()
        means1.reCluster()
        self.means = means1.calcSigma()
        self.inputNodes = [RbNode] * len(self.dataPoints)
        self.createInputNodes()
        self.createHiddenNodes()
        self.createOutNodes()
        self.calcPhiVals()
        self.calcOutValues()
        
        self.train(self.alpha)
    #Creates a node for each data point
    def createInputNodes(self):
        for i in range(len(self.dataPoints)):
            self.inputNodes[i] = RbNode.RbNode(self.dataPoints[i], self.expectedOutput[i], self.k)
    #Creates a node for each cluster
    def createHiddenNodes(self):
        print(len(self.means))
        for i in range(len(self.means)):
            self.hiddenNodes.append(RbNodeHidden.RbNodeHidden(self.means[i][0], self.means[i][1]))
    #Creates the specified number of output points
    def createOutNodes(self):
        for i in range(self.numOut):
            self.outputNodes.append(node.RBFNode(self.k))
    #Calculates the phi values for each input node
    def calcPhiVals(self):
        for i in range(len(self.inputNodes)):
            for j in range(len(self.hiddenNodes)):
                self.inputNodes[i].addPhi(self.hiddenNodes[j], j, len(self.dataPoints))
            self.inputNodes[i].phiValues.append(1)
            print(self.inputNodes[i].phiValues)
    #Sums the weights and corresponding phi values of each input to calculate an output      
    def calcOutValues(self):
        for i in range(len(self.inputNodes)):
            for j in range(len(self.outputNodes)):
                self.outputNodes[j].activeFunct(self.inputNodes[i] , j)
    #Runs
    def train(self, alpha):
        network = [[RbNode] * self.numOut] * 1
        for i in range(len(self.outputNodes)):
            network[0][i] = self.outputNodes[i]
    
        stop = False
        count = 0
        #Updates the weights for a specified number or iterations. 
        while stop != True:
            stop = True
            self.calcOutValues()
            for i in range(len(self.outputNodes)):
                
                self.errors.append(.5 * (math.pow(self.outputNodes[i].errorcount / (len(self.inputNodes)),2))) 
                self.outputNodes[i].errorcount = 0
            for i in range(len(self.outputNodes)):
                stop = self.outputNodes[i].updateWeights(self.alpha,len(self.inputNodes))
            count += 1
            if(count >50):
                stop = True
        print("rate", count)
        self.graphErrors(self.errors)
    def test(self, inputVectors, expectedOut):
        nodes = []
        testErrors = []
        for i in range(len(inputVectors)):
            testPhi = []
            node = RbNode.RbNode(inputVectors[i], expectedOut[i], self.k)
            for i in range(len(self.hiddenNodes)):
                testPhi.append(self.hiddenNodes[i].calcPhi(node.inputVector, len(inputVectors)))
            testPhi.append(1)
            node.phiValues = testPhi
            nodes.append(node)
        #Adds the error of each test point to 
        for j in range(len(self.outputNodes)):
            sumerrors = 0
            for i in range(len(nodes)):
                self.outputNodes[j].activeFunct(nodes[i], j)
                out = nodes[i].output[j]
                sumerrors += .5 * math.pow((expectedOut[i][0] - out), 2)
                testErrors.append(sumerrors / len(self.outputNodes))
        return testErrors
    #Graphs the error over time
    def graphErrors(self, error):
        error[0] = error[1]
        plt.plot(error)
        plt.show()
             
if __name__ == "__main__":    
    
    data1 = generate_data.GenerateData(1000, 6)
    data1.stratified_sample(10)
    input = data1.get_data()
    expected = data1.get_target_vector()
    data2 = generate_data.GenerateData(10, 2)
    data2.stratified_sample(10)
    test = data2.get_data()
    testOut = data2.get_target_vector()
    print(input)
    print(expected)
    rb1 = radialBasisOut(input, expected,250,1,.0005)
    rb2 = radialBasisOut(input, expected,250,1,.05)
    
    rb3 = radialBasisOut(input, expected,500,1,.0005)
    rb4 = radialBasisOut(input, expected,500,1,.05)
    rb5 = radialBasisOut(input, expected,750,1,.0005)
    #rb1.createNetwork()
    #rb2.createNetwork()
    #rb3.createNetwork()
    #rb4.createNetwork()
    rb5.createNetwork()
    #rb6.createNetwork()
    print(rb1.outputNodes[0].weights)
    errors = rb1.test(test, testOut)

    
    #print(errors)




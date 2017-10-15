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
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as gr
from matplotlib import pyplot as plt
from pandas.core.indexing import _IXIndexer



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
        #return self.test(testPoints, testExpectedOut)        
    #kmeans will return a means[] list, with coresponding tuples, with sigma
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
            #self.hiddenNodes.append(RbNodeHidden.RbNodeHidden(self.means[i][0], 5))
        print(len(self.hiddenNodes))
    def createOutNodes(self):
        for i in range(self.numOut):
            self.outputNodes.append(node.RBFNode(self.k))
    def calcPhiVals(self):
        for i in range(len(self.inputNodes)):
            for j in range(len(self.hiddenNodes)):
                self.inputNodes[i].addPhi(self.hiddenNodes[j], j)
            self.inputNodes[i].phiValues.append(1)
            print(self.inputNodes[i].phiValues)
            
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
            self.calcOutValues()
            for i in range(len(self.outputNodes)):
                
                self.errors.append(.5 * (math.pow(self.outputNodes[i].errorcount / (100000 * len(self.inputNodes)),2))) 
                #self.errors.append(self.outputNodes[i].errorcount/len(self.inputNodes))
                self.outputNodes[i].errorcount = 0
                #self.errors.append(self.outputNodes[i].errorcount / len(self.inputNodes))  
            for i in range(len(self.outputNodes)):
                stop = self.outputNodes[i].updateWeights(self.alpha,len(self.inputNodes))
                #if (self.outputNodes[i].maxError > 0.001):
                #   print(self.outputNodes[i].maxError, "maxE")
                 #   stop = False
            #self.calcOutValues()
            count += 1
            if(count >200):
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
                testPhi.append(self.hiddenNodes[i].calcPhi(node.inputVector))
            testPhi.append(1)
            node.phiValues = testPhi
            nodes.append(node)

        for j in range(len(self.outputNodes)):
            sumerrors = 0
            for i in range(len(nodes)):
                self.outputNodes[j].activeFunct(nodes[i], j)
                out = nodes[i].output[j]
                sumerrors += .5 * math.pow((expectedOut[i] - out), 2)
                print(out)
                print(expectedOut[i])
                print(out - expectedOut[i], "difference")
                print(.5 * math.pow((out - expectedOut[i]), 2), "meanSquared")
                print()
                testErrors.append(sumerrors / len(self.outputNodes))
        return testErrors
    def graphErrors(self, error):
        error[0] = error[1]
        plt.plot(error)
        plt.show()
                
if __name__ == "__main__":    
    data1 = generate_data.GenerateData(100, 2)
    data1.stratified_sample(10)
    input = data1.get_data()
    expected = data1.get_target_vector()
    data2 = generate_data.GenerateData(5, 2)
    data2.stratified_sample(10)
    test = data2.get_data()
    testOut = data2.get_target_vector()
    print(input)
    rb1 = radialBasisOut(input, expected,1,1,.005)
    rb1.createNetwork()
    print(rb1.outputNodes[0].weights)
    testErrors = rb1.test(test, testOut)
    print(testErrors)
    #rb1.graphErrors(testerrors)

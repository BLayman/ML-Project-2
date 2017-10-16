#!/usr/bin/env python3

from backprop.bpNetCreator import BPNetCreator
from shared.forwardProp import ForwardProp
from backprop.backProp import BackProp
from shared.gradientDescent import GradientDescent
from shared.printNetwork import NetworkPrinter
import random
from matplotlib import pyplot as plt

class BPAlg:

    def train(self, inputsArray, expectedOutputsArray, hiddenLayerNum, nodesInHLNum):
        plotErrors = []
        alpha = .0005
        convergenceEpsilon = .01
        regularizationParam = .1
        netPrinter = NetworkPrinter()
        netCreator = BPNetCreator(hiddenLayerNum,nodesInHLNum,len(inputsArray[0]),len(expectedOutputsArray[0]))
        network = netCreator.create()
        # print("------------- Post creation -----------")
        #netPrinter.printNet(network)
        stop = False
        counter = 0
        while(not stop):
            error = 0
            counter += 1
            print(counter)
            if (counter > 500): # 2000
                print("stopped early")
                break
            # forward propagate
            for i in range(len(inputsArray)):
                forwardProp = ForwardProp(network,inputsArray[i],expectedOutputsArray[i])
                #netPrinter.printNet(network)
                hypothesis = forwardProp.getHypothesis()
                #print("hypothesis: " + str(hypothesis))
                error = forwardProp.getTotalSquaredError()
                plotErrors.append(error)
                #print("**************           *****************  error  *********************: " + str(error))
                # back propagate
                BackProp(network)
                #print("------------- Post backward -----------")
                #netPrinter.printNet(network)
            # after batch learning, run gradient descent

            #print("Error: %f" % error)
            #plt.plot(counter, error / len(inputsArray[0]), 'ro')

            gradDesc = GradientDescent(network, alpha, len(inputsArray), regularizationParam, convergenceEpsilon)
            stop = gradDesc.updateWeights()
            #print("-------------------")
            #print("------------- Post Gradient Descent -----------")
            #netPrinter.printNet(network)
        print(stop)
        #plt.plot(plotErrors)
        #plt.show()
        return network
        
    def test(self, inputsArray, expectedOutputsArray, network):
        errors = []
        totalError = 0
        for i in range(len(inputsArray)):
            forwardProp = ForwardProp(network, inputsArray[i], expectedOutputsArray[i])
            error = forwardProp.getTotalSquaredError()
            errors.append(error)
        for error in errors:
            totalError += error
            #print("error: " + str(error))
        #print("total error: " + str(totalError))
        return errors


trainingXData = []
trainingYData = []
testDataX = []
testDataY = []

for i in range(50):
    x = random.uniform(0,10)
    print(x)
    trainingXData.append([x])
    trainingYData.append([x*x/x])
    
for i in range(10):
    x = random.uniform(0,10)
    testDataX.append([x])
    testDataY.append([x*x/x])


#test functionality
"""
bpAlg = BPAlg()
trainedNetwork = bpAlg.train(trainingXData,trainingYData, 2, 8)
bpAlg.test(testDataX, testDataY, trainedNetwork)
"""

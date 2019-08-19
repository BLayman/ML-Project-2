#!/usr/bin/env python3
import math

from backprop.bpNetCreator import BPNetCreator
from shared.forwardProp import ForwardProp
from backprop.backProp import BackProp
from shared.gradientDescent import GradientDescent
from shared.printNetwork import NetworkPrinter
import random
from matplotlib import pyplot as plt

class BPAlg:

    def train(self, inputsArray, expectedOutputsArray, hiddenLayerNum, nodesInHLNum, graphX, graphY):
        plotErrors = []
        alpha = .001
        convergenceEpsilon = .01
        regularizationParam = 0
        netPrinter = NetworkPrinter()
        netCreator = BPNetCreator(hiddenLayerNum,nodesInHLNum,len(inputsArray[0]),len(expectedOutputsArray[0]))
        network = netCreator.create()
        # print("------------- Post creation -----------")
        #netPrinter.printNet(network)
        stop = False
        counter = 0
        while(not stop):
            error = 0

            print(counter)
            if (counter > 7000): # 2000
                print("stopped early")
                break

            if counter % 100 == 0:
                self.graph(inputsArray, expectedOutputsArray, graphX, graphY, network, str(counter))
            # forward propagate
            for i in range(len(inputsArray)):
                forwardProp = ForwardProp(network,inputsArray[i],expectedOutputsArray[i])
                if counter == 0 and i == 0:
                    #netPrinter.printNet(network)
                    hypothesis = forwardProp.getHypothesis()
                    #print("actual: " + str(forwardProp.expectedOuts))
                    #print("hypothesis: " + str(hypothesis))
                error = abs(forwardProp.getHypothesis()[0] - forwardProp.expectedOuts[0])
                if counter % 100 == 0:
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
            #stop = False
            #print("-------------------")
            #print("------------- Post Gradient Descent -----------")
            #netPrinter.printNet(network)
            counter += 1

        print(stop)

        plt.clf()
        plt.plot(plotErrors)
        plt.savefig('./SavedFigures/error' + str(counter) + ".png")
        plt.show()

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
            print("error: " + str(error))
        print("total error: " + str(totalError))
        return errors


    def graph(self, inputsArray, expectedOutputsArray, testInputs, testOutputs, network, index):

        netOutputs = []

        for i in range(len(testInputs)):
            forwardPropTest = ForwardProp(network, testInputs[i], testOutputs[i])
            hypothesisTest = forwardPropTest.getHypothesis()
            netOutputs.append(hypothesisTest)


        plt.clf()
        plt.plot(testInputs, testOutputs)
        plt.plot(testInputs, netOutputs)
        plt.scatter(inputsArray, expectedOutputsArray, s=10, c="red")

        plt.savefig('./SavedFigures/plot' + index + ".png")


trainingXData = []
trainingYData = []
testDataX = []
testDataY = []

graphDataX = []
graphDataY = []

for i in range(30):
    x = random.uniform(-7,7)
    print(x)
    trainingXData.append([x/7])
    trainingYData.append([(x*x*x)])
    
for i in range(10):
    x = random.uniform(-7,7)
    testDataX.append([x/7])
    testDataY.append([(x*x*x)])

for i in range(200):
    x = (i - 100) / 10
    graphDataX.append([x/7])
    graphDataY.append([(x*x*x)])


#test functionality

bpAlg = BPAlg()
trainedNetwork = bpAlg.train(trainingXData,trainingYData, 2, 10, graphDataX, graphDataY)
# bpAlg.test(testDataX, testDataY, trainedNetwork)
# bpAlg.graph(graphDataX, graphDataY, trainedNetwork, 999)

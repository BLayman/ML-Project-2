from backprop.bpNetCreator import BPNetCreator
from shared.forwardProp import ForwardProp
from backprop.backProp import BackProp
from shared.gradientDescent import GradientDescent
from shared.printNetwork import NetworkPrinter

class BPAlg:
    
    def train(self, inputsArray, expectedOutputsArray, hiddenLayerNum, nodesInHLNum):
        alpha = .005
        convergenceEpsilon = .1
        regularizationParam = .1
        netPrinter = NetworkPrinter()
        netCreator = BPNetCreator(hiddenLayerNum,nodesInHLNum,len(inputsArray[0]),len(expectedOutputsArray[0]))
        network = netCreator.create()
        #print("------------- Post creation -----------")
        #netPrinter.printNet(network)
        stop = False
        counter = 0
        while(not stop):
            counter += 1
            if (counter > 7000):
                print("stopped early")
                break
            # forward propagate
            for i in range(len(inputsArray)):
                forwardProp = ForwardProp(network,inputsArray[i],expectedOutputsArray[i])
                #print("------------- Post forward -----------")
                #netPrinter.printNet(network)
                #hypothesis = forwardProp.getHypothesis()
                # print("hypothesis: " + str(hypothesis))
                error = forwardProp.getTotalMeanSquaredError()
                print("**************           *****************  error  *********************: " + str(error))
                # back propagate
                BackProp(network)
                #print("------------- Post backward -----------")
                #netPrinter.printNet(network)
            # after batch learning, run gradient descent
            gradDesc = GradientDescent(network, alpha, len(inputsArray), regularizationParam, convergenceEpsilon)
            stop = gradDesc.updateWeights()
            print("-------------------")
        #print("------------- Post Gradient Descent -----------")
        netPrinter.printNet(network)
        print(stop)
        return network
        
    def test(self, inputsArray, expectedOutputsArray, network):
        errors = []
        totalError = 0
        for i in range(len(inputsArray)):
            forwardProp = ForwardProp(network, inputsArray[i], expectedOutputsArray[i])
            error = forwardProp.getTotalMeanSquaredError()
            errors.append(error)
        for error in errors:
            totalError += error
            print("error: " + str(error))
        print("total error: " + str(totalError))
                

#test functionality
bpAlg = BPAlg()
trainedNetwork = bpAlg.train([[1],[3],[6],[10],[15]], [[2],[6],[12],[20],[30]], 2, 10)
bpAlg.test([[2],[5],[12]],[[4],[10],[24]],trainedNetwork)


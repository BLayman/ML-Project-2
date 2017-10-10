from backprop.bpNetCreator import BPNetCreator
from shared.forwardProp import ForwardProp
from backprop.backProp import BackProp
from shared.gradientDescent import GradientDescent
from shared.printNetwork import NetworkPrinter

class BPAlg:
    
    def train(self, inputsArray, expectedOutputsArray, hiddenLayerNum, nodesInHLNum):
        netPrinter = NetworkPrinter()
        netCreator = BPNetCreator(hiddenLayerNum,nodesInHLNum,len(inputsArray[0]),len(expectedOutputsArray[0]))
        network = netCreator.create()
        #print("------------- Post creation -----------")
        #netPrinter.printNet(network)
        stop = False
        counter = 0
        while(not stop):
            counter += 1
            if (counter > 1000):
                break
            # forward propagate
            for i in range(len(inputsArray)):
                forwardProp = ForwardProp(network,inputsArray[i],expectedOutputsArray[i])
                #print("------------- Post forward -----------")
                #netPrinter.printNet(network)
                hypothesis = forwardProp.getHypothesis()
                print("hypothesis: " + str(hypothesis))
                error = forwardProp.getTotalMeanSquaredError()
                print("**************           *****************  error  *********************: " + str(error))
                # back propagate
                BackProp(network)
                #print("------------- Post backward -----------")
                #netPrinter.printNet(network)
            # after batch learning, run gradient descent
            gradDesc = GradientDescent(network, .5, len(inputsArray))
            stop2 = gradDesc.updateWeights()
            print("------------- Post Gradient Descent -----------")
            netPrinter.printNet(network)
            print(stop)

#test functionality
bpAlg = BPAlg()
#bpAlg.train([[2,0],[0,2]], [[4],[2]], 2, 2)
bpAlg.train([[2,0],[0,2],[2,2],[4,4]], [[4],[2],[6],[20]], 2, 10)

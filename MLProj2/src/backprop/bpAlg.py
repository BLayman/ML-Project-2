from backprop.bpNetCreator import BPNetCreator
from shared.forwardProp import ForwardProp
from backprop.backProp import BackProp
from shared.gradientDescent import GradientDescent

class BPAlg:
    
    def train(self, inputsArray, expectedOutputsArray, hiddenLayerNum, nodesInHLNum):
        netCreator = BPNetCreator(hiddenLayerNum,nodesInHLNum,len(inputsArray[0]),len(expectedOutputsArray[0]))
        network = netCreator.create()
        stop = False
        counter = 0
        while(not stop):
            counter += 1
            if (counter > 10000):
                break
            # forward propagate
            for i in range(len(inputsArray)):
                forwardProp = ForwardProp(network,inputsArray[i],expectedOutputsArray[i])
                hypothesis = forwardProp.getHypothesis()
                print("hypothesis: " + str(hypothesis))
                errors = forwardProp.getTotalMeanSquaredError()
                print("error: " + str(errors))
                # back propagate
                BackProp(network)
            # after batch learning, run gradient descent
            gradDesc = GradientDescent(network, .1, len(inputsArray))
            stop = gradDesc.updateWeights()
            print(stop)

#test functionality
bpAlg = BPAlg()
bpAlg.train([[1],[2],[3]], [[1],[4],[9]], 2, 3)
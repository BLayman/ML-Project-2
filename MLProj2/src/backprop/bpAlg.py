from backprop.bpNetCreator import BPNetCreator
from shared.forwardProp import ForwardProp
from backprop.backProp import BackProp

class BPAlg:
    
    def train(self, inputs, expectedOutputs, hiddenLayerNum, nodesInHLNum):
        netCreator = BPNetCreator(hiddenLayerNum,nodesInHLNum,len(inputs),len(expectedOutputs))
        network = netCreator.create()
        forwardProp = ForwardProp(network,inputs,expectedOutputs)
        hypothesis = forwardProp.getHypothesis()
        print("hypothesis: " + str(hypothesis))
        errors = forwardProp.getErrorArray()
        print("subtraction errors: " + str(errors))
        # backprop = BackProp(errors, network)

#test functionality
bpAlg = BPAlg()
bpAlg.train([10], [.1, .2, .3], 2, 2)
from backprop.bpNetCreator import BPNetCreator
from shared.forwardProp import ForwardProp

class BPAlg:
    
    def __init__(self):
        self.test()
        
    def test(self):
        inputs = [10]
        expected = .1
        netCreator = BPNetCreator(2,2,len(inputs),1)
        network = netCreator.create()
        forwardProp = ForwardProp(network,inputs,expected)
        hypothesis = forwardProp.getHypothesis()
        print("hypothesis: " + str(hypothesis))
        error = forwardProp.getSubtractionError()
        print("subtraction error: " + str(error))

bpAlg = BPAlg()
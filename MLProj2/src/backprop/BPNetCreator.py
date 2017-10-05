from shared import node

class BPNetCreator:
    def __init__(self, inputs, hiddenLayerNum, nodesInHLNum, inNum, outNum):
        self.inputs
        self.hiddenLayerNum = hiddenLayerNum
        self.nodesInHLNum = nodesInHLNum
        self.inNum = inNum
        self.outNum = outNum
        self.network = []
        self.create()
       
    # create and return network of nodes
    def create(self):
        # hidden layers:
        # j represents layer
        for j in range(self.hiddenLayerNum - 1):
            # i represents ith node in layer
            for i in range(self.nodesInHLNum): 
                self.network[j][i] = node.BPNode(False);  
        # output layer node: uses no activation function, just sum
        self.network[self.hiddenLayerNum][0] = node.Node(True);
        return self.network
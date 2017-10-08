from shared import node

class BPNetCreator:
    def __init__(self, hiddenLayerNum, nodesInHLNum, inNum, outNum):
        self.hiddenLayerNum = hiddenLayerNum
        self.nodesInHLNum = nodesInHLNum
        self.inNum = inNum
        self.outNum = outNum
        self.network = []
       
    # create and return network of nodes
    def create(self):
        # hidden layers:
        # j represents layer
        for j in range(self.hiddenLayerNum):
            # create new column
            self.network.append([])
            # for every node in the layer
            for i in range(self.nodesInHLNum): 
                # if first hidden layer
                if (j == 0):
                    self.network[j].append(node.BPNode(self.inNum))
                # other hidden layers
                else:
                    self.network[j].append(node.BPNode(self.nodesInHLNum))
        # output layer node: uses no activation function, just sum
        self.network.append([])
        for i in range(self.outNum):
            self.network[self.hiddenLayerNum].append(node.Node(self.nodesInHLNum)) 
        return self.network
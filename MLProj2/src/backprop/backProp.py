class BackProp:
    
    def __init__(self, outputDeltas, network):
        self.outputDeltas = outputDeltas
        self.network = network
        self.backPropagate()
        self.accumulatePartials()
    
    # back propagate to get deltas
    def backPropagate(self):
        # for each hidden layer
        for j in reversed(range(len(self.network))):
            # for each neuron i, in layer j
            for i in range(len(self.network[j])):
                # error inherited from layer j + 1
                error = 0.0
                # for each neuron in layer j + 1
                for k in range(len(self.network[j+1])):
                    # take the ith index in that neurons weight array,
                    # and multiply it by that neuron's delta, then add that product to our error
                    error += self.network[j+1][k].getDelta() * self.network[j+1][k].getWeights()[i]
                # after error has been obtained, 
                # multiply it by the derivative of the activation function to get the next delta
                delta = error * self.activDeriv(self.network[j][i].getActiv())
                self.network[j][i].setDelta(delta)
    
    # then accumulate partial derivatives for each node 
    def accumulatePartials(self):
        # for the ith node in the jth layer
        for j in (range(len(self.network))):
            for i in (range(len(self.network[j]))):
                partials = []
                # for every node in the previous layer
                for k in range(len(self.network[j - 1])):
                    # the partial derivative for the weight connecting i in j to k in j -1 
                    partial = self.network[j][i].getDelta() * self.network[j-1][k].getActiv()
                    partials.append(partial)
                # accumulate list of partials in i of j ( because it contains the corresponding weights
                self.network[j][i].addPartials(partials)
                
    # derivative of activation function                  
    def activDeriv(self, activation):
        return activation * (1.0 - activation)
                
                    
                          
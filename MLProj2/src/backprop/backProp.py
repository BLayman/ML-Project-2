class BackProp:
    
    def __init__(self, outputDeltas, network):
        self.outputDeltas = outputDeltas
        self.network = network
        self.partials = [] # TODO: make partials an input argument
        
    def backpropagate(self):
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
                
    def accumulatePartials(self):
        for j in (range(len(self.network))):
            for i in (range(len(self.network[j]))):
                for k in (range(len(self.network[j+1]))):
                    self.partials[j][i][k] += self.network[j][i].getActiv() * self.network[j+1][k].getDelta()
                
                
    def activDeriv(self, activation):
        return activation * (1.0 - activation)
                
                    
                          
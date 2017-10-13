import math

class BackProp:
    
    def __init__(self, network):
        self.network = network
        self.backPropagate()
        self.accumulatePartials()
    
    # back propagate to get deltas
    def backPropagate(self):
        # for each hidden layer
        for j in reversed(range(1, len(self.network)-1)):
            # for each neuron i, in layer j
            for i in range(len(self.network[j])):
                # error inherited from layer j + 1
                error = 0.0
                # for each neuron k in layer j + 1 (starts as output layer)
                for k in range(len(self.network[j+1])-1):
                    # take the ith index in that neurons weight array,
                    # and multiply it by that neuron's delta, then add that product to our error
                    error += self.network[j+1][k].getWeights()[i] * self.network[j+1][k].getDelta()
                # if at last hidden layer, no hidden nodes to exclude at final node in layer
                if (j == len(self.network)-2):
                    error += self.network[j+1][len(self.network[j+1])-1].getWeights()[i] * self.network[j+1][len(self.network[j+1])-1].getDelta()
                # after error has been obtained, 
                # multiply it by the derivative of the activation function to get the next delta
                delta = error * self.activDeriv(self.network[j][i].getActiv())
                self.network[j][i].setDelta(delta)
    
    # then accumulate partial derivatives for each node 
    def accumulatePartials(self):
        # from the second hidden layer to the output
        for j in range(1, len(self.network)): 
            for i in range(len(self.network[j])):
                partials = [] # list of partial derivatives for weights contained in the ith node
                # for every weight in the nodes weight array
                for k in range(len(self.network[j][i].getWeights())):
                    # if in output layer, calculate special partial
                    if (j == len(self.network)-1):
                        partial = self.network[j][i].getErrorDiff() * self.network[j-1][k].getActiv()
                        partials.append(partial)
                    # if in hidden layers:
                    else:    
                        # the partial derivative for the weight connecting i in j to k in j -1 
                        partial = self.network[j][i].getDelta() * self.network[j-1][k].getActiv()
                        partials.append(partial)
                # accumulate list of partials in i of j ( because it contains the corresponding weights)
                self.network[j][i].addPartials(partials)
                
    # derivative of activation function                  
    def activDeriv(self, activation):
        return activation * (1.0 - activation)
                
                    
                          
import math
from _ast import If

class ForwardProp:
    def __init__(self, network, inputs, expectedOuts):
        self.expectedOuts = expectedOuts
        self.network = network
        self.inputs = inputs
        self.hypothesis = [] # will store list of outputs
        # calculate hypothesis
        self.calcHypothesis()
        
    def calcHypothesis(self):
        prevActivs = [] # list of activations from previous layer
        currentActivs = [] # to be used as previous activations in next iteration
        # first layer
        for i in range(len(self.network[0])):
            # initial activations are based on inputs
            self.network[0][i].setActiv(self.inputs)
            print("1st layer activ: " + str(self.network[0][i].getActiv()))
            prevActivs.append(self.network[0][i].getActiv())
        
        # other layers
        for j in range(1, len(self.network)):
            for i in range(len(self.network[j])):
                # set activations based on previous activations
                self.network[j][i].setActiv(prevActivs)
                print("hidden or output layer activ: " + str(self.network[j][i].getActiv()))
                # store activations in currentActivs list
                currentActivs.append(self.network[j][i].getActiv())
                # if we are in output layer, set output delta
                if (j == len(self.network)):
                    self.network[j][i].setDelta(self.network[j][i].getActiv() - self.expectedOuts[i])
            # prevActivs takes on values in currentActivs for next layer
            prevActivs = currentActivs
            currentActivs = []
        # outputs are final activations
        self.hypothesis = prevActivs
    
    # for use in test phase
    def getTotalMeanSquaredError(self):
        error = 0
        for i in range(len(self.expectedOuts)):
            error += math.pow((self.expectedOuts[i] - self.hypothesis[i]), 2)
        return error
    
    # for debugging (delta's already set for output layer)
    def getErrorArray (self):
        errors = []
        for i in range(len(self.expectedOuts)):
            # output deltas are simply difference between the hypothesis and the expected output
            errors.append(self.hypothesis[i] - self.expectedOuts[i]) 
        return errors
    
    # for debugging
    def getHypothesis(self):
        return self.hypothesis
            
            
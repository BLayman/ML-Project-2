import math
from shared.printNetwork import NetworkPrinter

class ForwardProp:
    def __init__(self, network, inputs, expectedOuts):
        self.expectedOuts = expectedOuts
        self.network = network
        # netPrinter = NetworkPrinter()
        # print(" -------------- INSIDE FORWARDPROP ---------------------")
        # netPrinter.printNet(network)
        self.inputs = inputs
        self.hypothesis = [] # will store list of outputs
        # calculate hypothesis
        self.calcHypothesis()
        
    def calcHypothesis(self):
        prevActivs = [] # list of activations from previous layer
        currentActivs = [] # to be used as previous activations in next iteration
        # input layer
        for i in range(len(self.network[0])-1):
            # initial activations are based on inputs
            self.network[0][i].setActiv(self.inputs[i])
            prevActivs.append(self.network[0][i].getActiv())
        # bias node: set activation to 1
        self.network[0][len(self.network[0]) - 1].setActiv(1)
        prevActivs.append(self.network[0][len(self.network[0]) - 1].getActiv())
        # hidden and output layers
        for j in range(1, len(self.network)):
            for i in range(len(self.network[j])):
                # set activations based on previous activations
                self.network[j][i].calcActiv(prevActivs)
                # store activations in currentActivs list
                currentActivs.append(self.network[j][i].getActiv())
                # if we are in output layer, set output delta
                if (j == len(self.network)-1):
                    #self.network[j][i].setErrorDiff(self.network[j][i].getActiv()-self.expectedOuts[i])
                    self.network[j][i].setDelta(self.calcError(self.network[j][i].getActiv(), self.expectedOuts[i]))
            # prevActivs takes on values in currentActivs for next layer
            prevActivs = currentActivs
            currentActivs = []
        # outputs are final activations
        self.hypothesis = prevActivs
        
    # calculated error given output and expected, used to calculate output deltas
    def calcError(self,output,expected):
        return output - expected
    
    # for use in test phase
    def getTotalSquaredError(self):
        error = 0
        for i in range(len(self.expectedOuts)):
            error += math.pow((self.expectedOuts[i] - self.hypothesis[i]), 2)
        return error

    # for use in test phase
    def getTotalError(self):
        error = 0
        for i in range(len(self.expectedOuts)):
            error += self.expectedOuts[i] - self.hypothesis[i]
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
            
            

import math

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
            # prevActivs takes on values in currentActivs for next layer
            prevActivs = currentActivs
            currentActivs = []
        # generate outputs
        self.hypothesis = prevActivs
    
            
    def getTotalMeanSquaredError(self):
        error = 0
        for i in range(len(self.expectedOuts)):
            error += math.pow((self.expectedOuts[i] - self.hypothesis[i]), 2)
        return error
    
    def getSubtractionErrorArray (self):
        errors = []
        for i in range(len(self.expectedOuts)):
            errors.append(self.expectedOuts[i] - self.hypothesis[i]) 
        return errors
    
    def getHypothesis(self):
        return self.hypothesis
            
            
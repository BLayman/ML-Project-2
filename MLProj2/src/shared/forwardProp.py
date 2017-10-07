import math

class ForwardProp:
    def __init__(self, network, inputs, expectedOut):
        self.expectedOut = expectedOut
        self.network = network
        self.inputs = inputs
        self.hypothesis = -1.0 # will store hypothesis
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
        
        self.hypothesis = self.network[len(self.network) - 1][0].getActiv()
    
            
    def getMeanSquaredError(self):
        return math.pow((self.hypothesis - self.expectedOut), 2)
    
    def getSubtractionError (self):
        return self.expectedOut - self.hypothesis
    
    def getHypothesis(self):
        return self.hypothesis
            
            
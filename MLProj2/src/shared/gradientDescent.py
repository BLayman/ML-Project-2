class GradientDescent:
    def __init__(self, network, alpha, dataSize):
        self.network = network
        self.dataSize = dataSize   
        self.alpha = alpha     
        
    def updateWeights(self):
        stop = True
        # for every node in the network
        for j in range(len(self.network)):
            for i in range(len(self.network[j])):
                self.network[j][i].updateWeights(self.alpha, self.dataSize)
                partials = self.network[j][i].getPartials()
                for par in partials:
                    if par > .1:
                        stop = False
        return stop
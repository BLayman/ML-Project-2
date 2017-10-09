class GradientDescent:
    def __init__(self, network, alpha, dataSize):
        self.network = network
        self.dataSize = dataSize        
        
    def updateWeights(self):
        stop = True
        # for every node in the network
        for j in self.network:
            for i in self.network[j]:
                self.network[i][j].updateWeights(self.alpha, self.dataSize)
                partials = self.network[i][j].getPartials()
                for par in partials:
                    if par > .1:
                        stop = False
        return stop
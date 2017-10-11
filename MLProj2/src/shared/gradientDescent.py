class GradientDescent:
    def __init__(self, network, alpha, dataSize, regParam, convergeEpsilon):
        self.network = network
        self.dataSize = dataSize   
        self.alpha = alpha 
        self.regParam = regParam
        self.convergeEpsilon = convergeEpsilon    
        
        
    def updateWeights(self):
        stop = True
        # for every node in the network
        for j in range(len(self.network)):
            for i in range(len(self.network[j])):
                self.network[j][i].updateWeights(self.alpha, self.dataSize, self.regParam)
                partials = self.network[j][i].getPartials()
                # every partial derivative must be less than .001 to conclude convergence 
                for par in partials:
                    if (par > self.convergeEpsilon or par < (-self.convergeEpsilon)):
                        stop = False
                self.network[j][i].setPartialsSum([])
        return stop
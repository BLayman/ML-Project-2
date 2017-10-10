from builtins import str
class NetworkPrinter:

    def printNet(self, network):
        for j in range(len(network)):
            print("layer: " + str(j) + " _________")
            for i in range(len(network[j])):
                print("Node: " + str(i) + " ----")
                node = network[j][i]
                self.printNode(node)  
                
    def printNode(self, node):
        print ("delta: " + str(node.getDelta()))
        print ("activ: " + str(node.getActiv()))
        print("weights: " + str(node.getWeights()))
        print("partials: " + str(node.getPartials()))
        #print("sum of partials: " + str(node.getParialsSum()))
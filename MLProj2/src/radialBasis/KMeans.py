import random
import math
from radialBasis import Cluster
from radialBasis import radialBasisOut

class KMeans:
    k = 0
    dataPoints = [[]]
    means = []
    meansIndex = []
    hasChanged = False
    def __init__(self, dataPoints, k):
        self.k = k
        self.dataPoints = dataPoints
        #Creates random indexes without replacement, for our initial centroids
        self.meansIndex = [None for c in range(self.k)]
        self.meansIndex = random.sample(range(0,len(self.dataPoints)), self.k)
        self.means = [Cluster for x in range(self.k)]
        #Creates a new cluster object for each random mean data point
        for i in range(len(self.meansIndex)):
            temporary = self.meansIndex[i]
            temp = Cluster.Cluster(self.dataPoints[temporary])
            self.means[i] = temp

    #Assigns 
    def calcMeans(self):

        for i in range(len(self.dataPoints)):
            distance = self.calcDistance(self.dataPoints[i], self.means[0].mean)
            minD = distance
            minClusterIndex = 0
            
            for j in range(1,len(self.means)):
                distance = self.calcDistance(self.dataPoints[i], self.means[j].mean)
                if distance < minD:
                    minD = distance
                    minClusterIndex = j

            tempCluster = self.means[minClusterIndex]
            if(len(tempCluster.clusterPoints[0]) == 0):
                tempCluster.clusterPoints[0] = self.dataPoints[i]
            else:
                tempCluster.clusterPoints.append(self.dataPoints[i])
            #self.means[minClusterIndex].addPoint(self.dataPoints[i])
        
    def reCluster(self):
        changed = True
        tempMeans = [[0 for i in range(len(self.means[0].mean))] for j in range(self.k)]
        for i in range(len(self.means) - 1):
            tempMeans[i] = self.means[i].mean
        
        while(changed == True):                  
            for i in self.means:
                i.calcCentroid()
            self.calcMeans()
            changed = False
            for i in range(len(self.means) - 1):
                for j in range(len(self.means[i].mean) - 1):
                    if (math.fabs(tempMeans[i][j] - self.means[i].mean[j])) > 0.00001:
                        changed = True
                tempMeans[i] = self.means[i].mean
    
    #Calculates the sigma value for each cluster based of the maximum pairwise distance within a cluster
    def calcSigma(self):
        meanWithSigma = [[] for i in range(len(self.means))]
        for i in range(len(self.means)):
            temp = self.means[i].clusterPoints
            tempList = []
            maxD = 0
            for j in range(len(temp)):
                for k in range(len(temp)):
                    d = self.calcDistance(temp[j], temp[k])
                    if d > maxD:
                        maxD = d
            print("max d", maxD)
            tempList.append(self.means[i].mean)
            sigma = float(maxD)/math.sqrt(2 * self.k)
            if (sigma < 0.00001):
                sigma = 0.0001
            tempList.append(sigma)
            print("sig" , sigma)
            meanWithSigma[i] = tempList
        return meanWithSigma
    '''
    def calcSigma(self):
        maxD = 0
        meanWithSigma = [[] for i in range(len(self.means))]
        for j in range(len(self.dataPoints)):
                for k in range(len(self.dataPoints)):
                    d = self.calcDistance(self.dataPoints[j], self.dataPoints[k])
                    if d > maxD:
                        maxD = d
        sigma = maxD / math.sqrt(len(self.means))
        for i in range(len(self.means)): 
            temp = self.means[i].clusterPoints
            tempList = []
            tempList.append(self.means[i].mean)
            tempList.append(sigma)
            meanWithSigma[i] = tempList
        return meanWithSigma
    '''
        
    #Calculates the distance in between two vectors
    def calcDistance(self, d1, d2):
        distance = 0.0
        for i in range(len(d1)):
            # adds the square of the difference of each variable in the data point
            distance += math.pow((d1[i] - (d2[i])),2) 
        #returns the Euclidean distance between two vectors
        return math.sqrt(distance)
#Main class for testing the K Means Algorithm
'''
if __name__ == "__main__":
    num = 5
    data = [[[0, None] for i in range(3)] for j in range(num)]

    for  i in range (len(data)):
        for j in range (len(data[i])):
            data[i][j] = random.randint(1,10)
    data1 = [[-1, -1], [-.9, -.9], [-.8, -.8], [-.7, -.7], [-.6, -.6], [-.5, -.5], [-.4, -.4], [-.3, -.3], [-.2, -.2],
             [-.1, -.1], [0, 0], [-1, 1], [-.9, .9], [-.8, .8], [-.7, .7], [-.6, .6], [-.5, .5], [-.4, .4],
             [-.3, .3], [-.2, .2], [-.1, .1], [1, -1], [.9, -.9], [.8, -.8], [.7, -.7], [.6, -.6], [.5, -.5], [.4, -.4], [.3, -.3],
             [.2, -.2], [.1, -.1], [1, 1], [.9, .9], [.8, .8], [.7, .7], [.6, .6], [.5, .5], [.4, .4], [.3, .3],
             [.2, .2], [.1, .1]]
    expectedOut = [0.0, 0.08099999999999996, 0.128, 0.14700000000000002, 0.14400000000000002, 0.125, 0.09600000000000002, 0.063, 0.03200000000000001, 0.009000000000000001, 0.0, 2.0, 1.5390000000000001, 1.1520000000000001, 0.8329999999999999, 0.576, 0.375, 0.22400000000000003, 0.11699999999999999, 0.04800000000000001, 0.011000000000000003, 0.0, 0.08099999999999996, 0.128, 0.14700000000000002, 0.14400000000000002, 0.125, 0.09600000000000002, 0.063, 0.03200000000000001, 0.009000000000000001, 2.0, 1.5390000000000001, 1.1520000000000001, 0.8329999999999999, 0.576, 0.375, 0.22400000000000003, 0.11699999999999999, 0.04800000000000001, 0.011000000000000003]


    means1 = KMeans(data1, 5)
    means1.calcMeans()
    means1.reCluster()
    means1Sigma = means1.calcSigma()
    radialBasis1 = radialBasisOut.radialBasisOut(data1, expectedOut, 5, 1,means1Sigma)
    '''
    


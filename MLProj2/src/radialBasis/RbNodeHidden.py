'''
Created on Oct 8, 2017

@author: Carsen
'''
import radialBasis
import math
class RbNodeHidden:
    sigma = 0
    mean = []
    
    def __init__(self, mean, sigma):
        self.sigma = sigma
        self.mean = mean
    #Calulates the phi value of a data point for a mean of a cluster
    def calcPhi(self,inputVector, sigma):
        distance = 0
        for i in range(len(inputVector)):
                # adds the square of the difference of each variable in the data point
                #distance += math.pow((inputVector[i] - self.mean[i]), 2)
                distance += math.pow((self.mean[i] - inputVector[i] ), 2)
        phi = math.exp(-distance/(sigma))
        #phi = math.exp(-distance/(2 * self.sigma * self.sigma))
        return phi
            
    
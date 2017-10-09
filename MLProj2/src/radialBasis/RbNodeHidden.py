'''
Created on Oct 8, 2017

@author: Carsen
'''
import radialBasis
import math
class RbNodeHidden:
    sigma = 0
    mean = []
    
    def __init__(self, sigma, mean):
        self.sigma = sigma
        self.mean = mean
    
    def calcPhi(self,inputVector):
        distance = 0
        for i in inputVector:
                # adds the square of the difference of each variable in the data point
                distance += math.pow((inputVector[i] - self.mean[i]), 2) 
        phi = math.exp(-distance/(2 * self.sigma * self.sigma))
        return phi
            
    
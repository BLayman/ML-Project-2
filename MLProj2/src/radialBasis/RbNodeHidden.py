'''
Created on Oct 8, 2017

@author: Carsen
'''
import radialBasis
import math
sigma = 0
mean = []

def __init__(self, sigma, mean):
    self.sigma = sigma
    self.mean = mean

def calcPhi(inputVector):
    distance = 0
    for i in inputVector:
            # adds the square of the difference of each variable in the data point
            distance += (inputVector[i] - mean[i])^2 
    phi = math.exp(-distance/(2 * sigma * sigma))
    return phi
        
    
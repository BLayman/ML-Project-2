'''
Created on Oct 8, 2017

@author: Carsen
'''
class RbNode:
    inputVector = []
    k = 0
    expectedOut = 0
    output = [0] * k
    phiValues = []
    
    def __init__(self, inputVector, expectedOut, k):
        self.inputVector = inputVector
        self.expectedOut = expectedOut
        self.phiValues = [0] * k
        self.k = k 
    
    #Adds the phi value for the input vector in comparison to a mean of a cluster. 
    def addPhi(self, rbNodeHid,j, l):
        self.phiValues[j] = rbNodeHid.calcPhi(self.inputVector, l)
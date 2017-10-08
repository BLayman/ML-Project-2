'''
Created on Oct 8, 2017

@author: Carsen
'''
inputVector = []
expectedOut = 0
output = []
phiValues = []

def __init__(self, inputVector, expectedOut):
    self.inputVector = inputVector
    self.expectedOut = expectedOut

def addPhi(self, rbNodeHid):
    phiValues.append(rbNodeHid.calcPhi(self.inputVector))
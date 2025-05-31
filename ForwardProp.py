import numpy as np
import pandas as pd


#only 1 hidden layer.
def initializeParams (input_size = 784, hidden_size = 128, output = 10):
    # Use He initialization for better gradient flow
    W1 = np.random.randn(input_size, hidden_size) * 0.01

    #bias from input to hidden layer
    #starting with 0
    B1 = np.zeros((hidden_size, 1)) *0.01
    
    #weights from hidden to output
    W2 = np.random.randn(hidden_size, output) * 0.01

    #bais from hidden to output
    #starting with 0
    B2 = np.zeros((output,1 )) *0.01

    return W1, B1, W2, B2

def ReLu(Z):
    return np.maximum(0,Z)

#returns a probability
def softMax(Z):
    expZ = np.exp(Z-np.max(Z, axis = 0, keepdims= True))

    #this is vectorized like it would be in R code. 
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forwardProp(x, W1, B1, W2, B2):
    Z1 =  W1.T @ x + B1
    hidden_layer = ReLu(Z1)
    
    Z2 = W2.T @ hidden_layer + B2

    #print(Z2)

    output = softMax(Z2)

    return Z1, hidden_layer, Z2, output
    

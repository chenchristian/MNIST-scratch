import numpy as np
import pandas as pd


#only 1 hidden layer.
def initializeParams (input_size = 784,  hidden_size = 128, output = 10):
    #weights from input to hidden layer
    W1 = np.random.randn(input_size, hidden_size)

    #bias from input to hidden layer
    B1 = np.random.randn(hidden_size, 1)
    
    #weights from hidden to output
    W2 = np.random.randn(hidden_size, output)

    #bais from hidden to output
    B2 = np.random.randn(output, 1)

    return W1, B1, W2, B2

def ReLu(Z):
    return np.maximum(0,Z)

#returns a probability
def softMax(Z):
    expZ = np.exp(Z-np.max(Z, axis = 0, keepdims= True))

    #this is vectorized like it would be in R code. 
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forwardProp(x, W1, B1, W2, B2):
    Z =  W1.T @ x + B1
    hidden_layer = ReLu(Z)
    
    Z2 = W2.T @ hidden_layer + B2

    print(Z2)

    output = softMax(Z2)

    return(output)
    

x = pd.read_csv("/Users/christianchen/VSCode_Python/Stat21/mnist_training.csv", header = None)



#gets the first row, and everything but first column
row = 1
image = x.iloc[row, 1:].values.reshape(-1, 1)
answer = x.iloc[row,1]


W1, B1, W2, B2 = initializeParams()

output = forwardProp(image, W1, B1, W2, B2)


rounded_output = np.round(output, 5)

print(rounded_output)
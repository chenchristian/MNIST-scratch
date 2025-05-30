import numpy as np
import pandas as pd

def backwardProp(x, y, Z1, A1, A2, W2):
    """
    Uses cached forward pass values to compute gradients.
    
    Args:
        x: Input vector (784, 1)
        y: One-hot label (10, 1)
        Z1: Linear output before ReLU (128, 1)
        A1: Activation of hidden layer (128, 1)
        A2: Output from softmax (10, 1)
        W2: Weight matrix from hidden to output (128, 10)
    
    Returns:
        dW1, dB1, dW2, dB2
    """
    dZ2 = A2 - y                # (10, 1)
    dW2 = A1 @ dZ2.T            # (128, 1) @ (1, 10) = (128, 10)
    dB2 = dZ2                  # (10, 1)

    dA1 = W2 @ dZ2              # (128, 1)
    dZ1 = dA1 * (Z1 > 0)        # ReLU derivative
    dW1 = x @ dZ1.T             # (784, 1) @ (1, 128) = (784, 128)
    dB1 = dZ1                   # (128, 1)

    return dW1, dB1, dW2, dB2

import numpy as np
import pandas as pd
from ForwardProp import forwardProp

# Load MNIST test data
test_data = pd.read_csv("/Users/christianchen/VSCode_Python/Stat21/mnist_training.csv", header=None)
x_test = test_data.iloc[:, 1:].values  # Features
y_test = test_data.iloc[:, 0].values   # Labels

    # Initialize counters
correct = 0
total = len(x_test)

weights = np.load("/Users/christianchen/VSCode_Python/Stat21/trained_weights.npz")
W1 = weights["W1"]
B1 = weights["B1"]
W2 = weights["W2"]
B2 = weights["B2"]


# Test the model
for i in range(total):
    # Get input and reshape
    x = x_test[i].reshape(-1, 1)          # (784, 1)
    true_label = int(y_test[i])
    
    # Forward pass only
    Z1, A1, Z2, A2 = forwardProp(x, W1, B1, W2, B2)
    
    # Get prediction (index of highest probability)
    predicted_label = np.argmax(A2)
    
    if predicted_label == true_label:
        correct += 1

accuracy = correct / total * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

print("A2:", A2.ravel())
print("y_label:", true_label)
print("A2[y_label][0]:", A2[true_label][0])
print("correct:", correct)


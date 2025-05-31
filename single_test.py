import numpy as np
from PIL import Image
from ForwardProp import forwardProp
import time
import os

def predict_single_image(x, label, return_probs=False):

    
    BASE_DIR = os.path.dirname(__file__)
    weights_path = os.path.join(BASE_DIR, "fine_tuned_weights.npz")

    # Load trained weights
    weights = np.load(weights_path)
    W1 = weights["W1"]
    B1 = weights["B1"]
    W2 = weights["W2"]
    B2 = weights["B2"]
    


    # Forward pass
    Z1, A1, Z2, A2 = forwardProp(x, W1, B1, W2, B2)


    # Get prediction
    predicted_label = np.argmax(A2)

    
    # Print results
    print(f"\nPredicted digit: {predicted_label}")
    print(f"True digit: {label}")
    print(f"Correct: {predicted_label == label}")
    
    # Print probabilities for all digits
    print("\nProbabilities for each digit:")
    for digit, prob in enumerate(A2.ravel()):
        print(f"Digit {digit}: {prob*100:.2f}%")

    if return_probs:
        return A2.ravel()
    return predicted_label

if __name__ == "__main__":
    # Replace these paths with your image and label paths
    image_path = "/Users/christianchen/VSCode_Python/Stat21/mnist_png/my_test/temp_drawing.png"
    label = 1
    
    predict_single_image(image_path, label)

import numpy as np
from PIL import Image
from ForwardProp import forwardProp
import time
import os

def predict_single_image(image_path, label, return_probs=False):

    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 for MNIST
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = (img_array > 128).astype(int)  # Binarize like in your training data
    
    
    # Reshape to match your model's input format
    x = img_array.reshape(-1, 1)  # (784, 1)

    


    # Load trained weights
    weights = np.load("/Users/christianchen/VSCode_Python/Stat21/fine_tuned_weights.npz")
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

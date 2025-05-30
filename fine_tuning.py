import numpy as np
import pandas as pd
import time
from ForwardProp import forwardProp
from BackProp import backwardProp
from data_cleaning import flatten_image_with_labels

# Path to fine tuning data
fine_tuning_path = "/Users/christianchen/VSCode_Python/Stat21/mnist_png/augmented"

# Convert images to matrix format
print("Converting fine-tuning images to matrix format...")
matrix = flatten_image_with_labels(fine_tuning_path, "fine_tuning_data.csv")

# Load the matrix data
data = pd.read_csv("fine_tuning_data.csv", header=None)
x_data = data.iloc[:, 1:].values  # Features
y_data = data.iloc[:, 0].values   # Labels

# Load initial weights
print("Loading initial weights...")
weights = np.load("/Users/christianchen/VSCode_Python/Stat21/95percent_15epoch_01alpha_.npz")
W1 = weights["W1"] 
B1 = weights["B1"]
W2 = weights["W2"]
B2 = weights["B2"]

def one_hot_encode(y, num_classes=10):
    vec = np.zeros((num_classes, 1))
    vec[y] = 1
    return vec

def fine_tuning(x_data, y_data, W1, B1, W2, B2,
                num_epochs=10, learning_rate=0.01, batch_size=32):
    #using previously trained weights
    W1, B1, W2, B2
    num_samples = len(x_data)
    num_batches = num_samples // batch_size

    loss_history = []

    time_start = time.time()

    for epoch in range(num_epochs):
        total_loss = 0
        
        # Shuffle the data at the start of each epoch
        indices = np.random.permutation(num_samples)
        x_shuffled = x_data[indices]
        y_shuffled = y_data[indices]

        for batch in range(num_batches):
            # Get batch data
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Initialize batch gradients
            batch_dW1 = np.zeros_like(W1)
            batch_dB1 = np.zeros_like(B1)
            batch_dW2 = np.zeros_like(W2)
            batch_dB2 = np.zeros_like(B2)
            batch_loss = 0

            # Process each sample in the batch
            for i in range(batch_size):
                # Get input and reshape
                x = x_batch[i].reshape(-1, 1)          # (784, 1)
                y_label = int(y_batch[i])
                y = one_hot_encode(y_label)          # (10, 1)

                # Forward pass
                Z1, A1, Z2, A2 = forwardProp(x, W1, B1, W2, B2)

                # Loss
                loss = -np.log(A2[y_label][0] + 1e-8)  # Added epsilon
                batch_loss += loss

                # Backward pass
                dW1, dB1, dW2, dB2 = backwardProp(x, y, Z1, A1, A2, W2)

                # Accumulate gradients
                batch_dW1 += dW1
                batch_dB1 += dB1
                batch_dW2 += dW2
                batch_dB2 += dB2

            # Average the gradients over the batch
            batch_dW1 /= batch_size
            batch_dB1 /= batch_size
            batch_dW2 /= batch_size
            batch_dB2 /= batch_size

            # Update weights
            W1 -= learning_rate * batch_dW1
            B1 -= learning_rate * batch_dB1
            W2 -= learning_rate * batch_dW2
            B2 -= learning_rate * batch_dB2

            total_loss += batch_loss

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_samples
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"Total time taken: {end_time - time_start:.2f} seconds")


    return W1, B1, W2, B2

if __name__ == "__main__":
    # Train on fine-tuning data
    print("Fine-tuning model...")
    W1, B1, W2, B2 = fine_tuning(x_data, y_data, W1, B1, W2, B2, num_epochs=30, learning_rate=0.01, batch_size=32)

    # Save fine-tuned weights
    print("Saving fine-tuned weights...")
    np.savez("fine_tuned_weights.npz", W1=W1, B1=B1, W2=W2, B2=B2)

    print("Fine-tuning complete! Weights saved to fine_tuned_weights.npz")




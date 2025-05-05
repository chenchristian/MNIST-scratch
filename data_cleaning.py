import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import struct
from array import array
from os.path  import join
from PIL import Image
import numpy as np
import os


"""
Purpose of this file was to read in the 28x28 images and convert them into a csv attaching a label. 
The data is stored in an n by 785 matrix with the first column being the label and the other columns
representing the 784 pixels. 

For this experiment each pixel is either 1 or 0. 
"""



def flatten_image_with_labels(folder_path, output_csv="labeled_mnist.csv"):
    """
    Flattens images and prepends their label (from folder name) to each row.
    Saves an (n x 785) CSV: [label, pixel1, ..., pixel784]
    """
    rows = []

    for label in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label)

        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if not filename.endswith(".png"):
                continue

            img_path = os.path.join(label_path, filename)
            try:
                img = Image.open(img_path).convert("L").resize((28, 28))
                img_array = np.array(img).flatten()
                img_vector = (img_array > 0).astype(int)  # binarize
                row = np.insert(img_vector, 0, int(label))  # prepend label
                rows.append(row)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    full_matrix = np.array(rows)

    # Save to CSV with label as first column
    np.savetxt(output_csv, full_matrix, delimiter=",", fmt="%d")
    print(f"Saved labeled matrix of shape {full_matrix.shape} to {output_csv}")

    return full_matrix


def csv_to_matrix(file_path):
    """
    Loads a CSV file, and returns it as a NumPy matrix.
    
    Assumes the file has no header and each row is: label, pixel1, ..., pixel784.
    """
    try:
        df = pd.read_csv(file_path, header=None)
        #print("CSV Head:\n", df.head())

        matrix = df.to_numpy()
        #print("\nMatrix shape:", matrix.shape)

        return matrix
    except Exception as e:
        #print(f"Error reading {file_path}: {e}")
        return None


#base_dir = "/Users/christianchen/VSCode_Python/Stat21/mnist_png/testing"
#flatten_image_with_labels(base_dir, output_csv="mnist_testing.csv")

#training_matrix = csv_to_matrix("/Users/christianchen/VSCode_Python/Stat21/mnist_training.csv")
testing_matrix = csv_to_matrix("/Users/christianchen/VSCode_Python/Stat21/mnist_testing.csv")

print(testing_matrix)
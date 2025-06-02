import numpy as np
from PIL import Image
from ForwardProp import forwardProp
import time
import streamlit as st
import io
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import os
import random
from single_test import predict_single_image
from streamlit_option_menu import option_menu
from pathlib import Path

#

#def save_to_fine_tuning(img, label):
    # Create directory if it doesn't exist
    #label_dir = os.path.join(FINE_TUNING_DIR, str(label))
    #os.makedirs(label_dir, exist_ok=True)
    
    # Generate random number between 1 and 5000
    #random_num = random.randint(1, 5000)
    #save_path = os.path.join(label_dir, f"{random_num}.png")
    
    # Save the image
    #img.save(save_path)
    #return save_path

def _clear_canvas():
        st.session_state.canvas_key += 1
# ←── NAVIGATION ──────────────────────────────────────────────
page = st.sidebar.radio("Go to", ["Play", "How it works"])

if page == "Play":
    st.title("Draw a Digit, We'll Predict it!")
    st.write("""
    Welcome to this interactive digit recognition demo! This application uses a neural network 
    to recognize handwritten digits.
    
    Draw any digit (0-9) in the canvas on the left, and the model will predict which digit it is. 
    The bar chart on the right shows the model's confidence for each possible digit.
             
    Tips: Try and fill the canvas all the way. If the drawing is too small, the model will not be able to predict it correctly.    
    """)

    if st.button("How it works"):
        page = "How it works"

    # Add label input
    #label = st.number_input("Enter the digit label (0-9):", min_value=0, max_value=9, value=0, step=1)

    # Create two columns for layout
    col1, col2 = st.columns(2)

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    #canvas to draw
    with col1:

        
        # Create a canvas component
        canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
       
    )
        
   

    #bar chart to show the probabilities
    with col2:
        # create a placeholder instead of a permanent chart
        chart_placeholder = st.empty()
        # draw an initial empty chart
        empty_df = pd.DataFrame({'Probability': [0.0]*10}, index=range(10))
        chart_placeholder.bar_chart(empty_df)

    # When the user clicks the predict button
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # Convert canvas to image
            img = Image.fromarray(canvas_result.image_data).convert("L")
            img_small = img.resize((28, 28))
            img_array = (np.array(img_small) > 128).astype(int)
        
            # Reshape to match your model's input format
            x = img_array.reshape(-1, 1)  # (784, 1)
        
            # Save to fine-tuning directory
            #save_path = save_to_fine_tuning(img_small, label)
            #st.write(f"Saved to fine-tuning directory: {save_path}")
        
        # Get probabilities and update bar chart
        probs = predict_single_image(x, None, return_probs=True)
        # Make prediction
        predicted_label = int(np.argmax(probs))
        
        # Display prediction
        st.write(f"Predicted Digit: {predicted_label}")
        
       
        
        df = pd.DataFrame({'Probability': probs}, index=range(10))
        # **replace** the chart
        chart_placeholder.bar_chart(df)

    # run this command to run your terminal
    # streamlit run /Users/christianchen/VSCode_Python/Stat21/streamlit.py

     # Add a clear button next to predict
    st.button("Clear", on_click=_clear_canvas)
      



if page == "How it works":
    st.title("How it works")

    st.markdown(
        """
        You can find all of the source files for this project on GitHub:

    👉 [MNIST-scratch repository](https://github.com/chenchristian/MNIST-scratch)
    """,
    unsafe_allow_html=True)

    # Embed YouTube video explaining neural networks
    st.video("https://youtu.be/t6-I3Ta3TNQ")
    
    st.markdown("""
    ### Building a Neural Network for MNIST Digit Recognition
    
    This project implements a feedforward neural network from scratch to recognize handwritten digits. The network architecture consists of:
    
    - Input layer (784 neurons - flattened 28x28 pixel images)
    - Hidden layer (128 neurons with ReLU activation)
    - Output layer (10 neurons with Softmax activation)
    
    The implementation includes:
    
    1. Forward propagation with ReLU and Softmax activation functions
    2. Backward propagation to compute gradients
    3. Mini-batch gradient descent optimization
    4. Cross-entropy loss function
    
    The model is trained on the MNIST dataset using batches of 32 images over 15 epochs. 
    
    The interactive demo above lets you draw digits and see the network's predictions in real-time, demonstrating how the trained model generalizes to new handwritten inputs.
    """)








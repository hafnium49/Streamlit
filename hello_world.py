# cnn_app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Define a function to preprocess the image to make it suitable for prediction with MobileNetV2
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    # Convert image to the RGB format
    image = image[:, :, :3]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title("Image Classifier using MobileNetV2")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and get predictions
    preprocessed = preprocess_image(image)
    predictions = model.predict(preprocessed)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())

    # Display the top prediction
    category, description, confidence = decoded_predictions[0][0]
    st.write(f"Prediction: {description} ({confidence:.2f}%)")

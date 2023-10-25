# streamlit_app.py

import streamlit as st
from PIL import Image

def to_gray(img):
    return img.convert("L")

st.title("Grayscale Image Converter")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Processing...")

    gray_image = to_gray(image)
    
    st.image(gray_image, caption="Grayscaled Image.", use_column_width=True)

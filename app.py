import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils.preprocessing import preprocess_image
import os

st.set_page_config(page_title="Medical Image Diagnosis", layout="wide")

st.title("🩺 Medical Disease Diagnosis using CNN")
st.markdown("Upload a medical scan (e.g., Chest X-ray) to get a diagnostic prediction and a Grad-CAM visualization of the model's focus.")

# Ensure a models directory exists, even if empty
os.makedirs("saved_models", exist_ok=True)

uploaded_file = st.file_uploader("Upload Medical Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Scan")
        image = Image.open(temp_path)
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Analysis Results")
        st.warning("⚠️ **Clinical Disclaimer**: This tool is for research purposes. A clinical diagnosis must be made by a certified medical professional.")
        
        st.info("Pre-trained model loading placeholder (Simulating prediction...)")
        
        # Simulate prediction logic (In a real app, model.predict() would be here)
        prediction_score = np.random.uniform(0.7, 0.99)
        class_name = "Positive (Abnormal)" if prediction_score > 0.5 else "Negative (Normal)"
        
        st.metric(label="Predicted Class", value=class_name)
        st.metric(label="Confidence", value=f"{prediction_score*100:.2f}%")
        
        st.subheader("Grad-CAM Explainability")
        st.text("Heatmap will highlight the regions the model focused on.")
        st.image(image, caption="Grad-CAM Overlay (Simulation)", use_column_width=True)

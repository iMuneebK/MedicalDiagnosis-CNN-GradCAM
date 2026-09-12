# Medical Disease Diagnosis CNN

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

A Convolutional Neural Network (CNN) based system designed to assist in the diagnosis of medical conditions from imaging data such as chest X-rays, skin lesions, and retinal scans.

## ⚠️ Medical Disclaimer
**This software is intended for educational and research purposes only.** It is not intended for use in the diagnosis of disease or other conditions, or in the cure, mitigation, treatment, or prevention of disease. Always consult a qualified healthcare provider for medical advice.

## Features
- **Deep Learning Architectures**: Includes both a custom CNN model and a Transfer Learning model using `ResNet50`.
- **Robust Training Pipeline**: Automated data augmentation, learning rate scheduling, and early stopping to prevent overfitting.
- **Explainability (Grad-CAM)**: Visualizes the regions of the image that contributed most to the model's decision, providing transparency for clinical interpretation.
- **Web Interface**: A clean Streamlit application for image upload, inference, and visualization.

## Dataset Information
This project is designed to be compatible with standard medical datasets (e.g., RSNA Pneumonia Detection Challenge, NIH Chest X-rays). Place your datasets in a `data/train` and `data/val` directory structure.

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Architecture
The transfer learning pipeline utilizes `ResNet50` pre-trained on ImageNet. The base layers are initially frozen, and a custom classification head (Global Average Pooling -> Dense -> Dropout -> Output) is trained on the specific medical dataset.

## Explainability (Grad-CAM)
Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

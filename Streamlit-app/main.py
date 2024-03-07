import streamlit as st
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
from pathlib import Path

from utils import *

def main():
    st.title("Nepali Consonant Hand Sign Classifier")

    # Setting device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # st.write(f"Device: {device}")

    # Load Trained Model
    MODEL_SAVE_PATH = "../NepaliHandSignClassifier.pth"
    model_info = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))

    # Instantiate the EfficientNet model
    model = EfficientNet.from_pretrained('efficientnet-b0').to(device)
    model._fc = nn.Linear(model._fc.in_features, 36)  

    # Define paths
    data_path = Path("test_images/")

    # Image upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image
        custom_image_path = data_path / uploaded_file.name
        with open(custom_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and preprocess the image
        custom_image_transformed = load_and_preprocess_image(custom_image_path)

        # Load the model
        model.load_state_dict(model_info)
        model.eval()

        # Predict the label for the image
        class_names = np.array(['क', 'क्ष', 'ख', 'ग', 'घ', 'ङ', 'च', 
                                'छ', 'ज', 'ज्ञ', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ',
                                'ण', 'त', 'त्र', 'थ', 'द', 'ध', 'न', 'प', 'फ',
                                'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष',
                                'स', 'ह'])
        predicted_label, image_pred_probs = predict_image(model,
                                                          custom_image_transformed,
                                                          class_names)


        # Prediction result section
        st.markdown(
            f'<h3 style="color: green;">Prediction Result</h3>', 
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 3])

        # Display prediction label and confidence rate on the left column
        col1.write(f"Predicted Neapli Consonant: **{predicted_label[0]}**")
        col1.write(f"Confidence: **{image_pred_probs.max()* 100:.2f}%**")

        # Display the uploaded image on the right column
        with col2:
            image = Image.open(custom_image_path)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
if __name__ == "__main__":
    main()

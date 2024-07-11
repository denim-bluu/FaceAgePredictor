import streamlit as st
from PIL import Image

from src.prediction.predict import predict_age


def image_prediction_page(model, face_detector):
    uploaded_files = st.file_uploader(
        "Choose image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            face_image = face_detector.detect_face(image)
            if face_image is not None:
                st.image(face_image, caption="Detected Face", use_column_width=True)
                age = predict_age(model, face_image)
                st.success(f"Predicted Age: {age:.2f} years")
            else:
                st.error("No face detected in the image.")

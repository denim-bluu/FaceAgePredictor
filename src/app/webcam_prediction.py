import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.prediction.predict import predict_age


def webcam_prediction_page(model, face_detector):
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        face_image = face_detector.detect_face(pil_image)

        if face_image is not None:
            age = predict_age(model, face_image)
            frame = np.array(face_image)
            frame = cv2.putText(
                frame,
                f"Age: {age:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        FRAME_WINDOW.image(frame)
    else:
        st.write("Stopped")

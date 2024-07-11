import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.prediction.predict import predict_age


def video_prediction_page(model, face_detector):
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        vf = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        age_predictions = []

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            face_image = face_detector.detect_face(pil_image)

            if face_image is not None:
                age = predict_age(model, face_image)
                age_predictions.append(age)
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

            stframe.image(frame)

        vf.release()

        if age_predictions:
            avg_age = np.mean(age_predictions)
            st.success(f"Average Predicted Age: {avg_age:.2f} years")
            st.line_chart(age_predictions)
        else:
            st.error("No faces detected in the video.")

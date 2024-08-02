import tempfile

import imageio
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from faceagepredictor.prediction.predict import predict_age


def video_prediction_page(model, face_detector):
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        # Open the video file using imageio
        vf = imageio.get_reader(tfile.name)

        stframe = st.empty()
        age_predictions = []

        for i in range(vf.get_length()):
            frame = vf.get_data(i)
            frame = frame[:, :, :3]  # Ensure the frame is in RGB format
            pil_image = Image.fromarray(frame)
            face_image = face_detector.detect_face(pil_image)

            if face_image is not None:
                age = predict_age(model, face_image)
                age_predictions.append(age)
                draw = ImageDraw.Draw(face_image)
                font = ImageFont.load_default()
                draw.text((10, 30), f"Age: {age:.2f}", fill=(0, 255, 0), font=font)
                frame = np.array(face_image)

            stframe.image(frame)

        vf.close()

        if age_predictions:
            avg_age = np.mean(age_predictions)
            st.success(f"Average Predicted Age: {avg_age:.2f} years")
            st.line_chart(age_predictions)
        else:
            st.error("No faces detected in the video.")

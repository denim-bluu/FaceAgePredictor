import streamlit as st
import torch

from src.data.face_detection import FaceDetector
from src.models.resnet_model import ResNetAgePredictor

from src.app.image_prediction import image_prediction_page
from src.app.video_prediction import video_prediction_page
from src.app.webcam_prediction import webcam_prediction_page

from src.utils.load_config import load_config


@st.cache_resource
def load_model():
    config = load_config()
    model = ResNetAgePredictor()
    model.load_state_dict(torch.load(config["paths"]["model_path"]))
    model.eval()
    return model


@st.cache_resource
def load_face_detector():
    return FaceDetector()


def main():
    st.title("Age Prediction App")

    model = load_model()
    face_detector = load_face_detector()

    upload_type = st.radio("Choose upload type:", ("Image", "Video", "Use Webcam"))

    if upload_type == "Image":
        image_prediction_page(model, face_detector)
    elif upload_type == "Video":
        video_prediction_page(model, face_detector)
    elif upload_type == "Use Webcam":
        webcam_prediction_page(model, face_detector)

    st.write(
        "Note: This model provides estimates based on visual appearance and may not always accurately reflect a person's true age."
    )


if __name__ == "__main__":
    main()

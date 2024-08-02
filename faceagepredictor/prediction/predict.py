import torch
from PIL import Image
from torch import nn

from faceagepredictor.data.data_prep import get_transforms
from faceagepredictor.data.face_detection import FaceDetector
from faceagepredictor.models.resnet_model import (
    ResNetAgePredictor,
)


def preprocess_image(image: Image.Image):
    transform = get_transforms()
    return torch.tensor(transform(image)).unsqueeze(0)


def load_model(model_path: str):
    model = ResNetAgePredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_age(model: nn.Module, face_image: Image.Image):
    input_tensor = preprocess_image(face_image)
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.item()


def predict_from_image(image_path: str, model: nn.Module, face_detector: FaceDetector):
    image = Image.open(image_path).convert("RGB")
    face_image = face_detector.detect_face(image)

    if face_image is None:
        return "No face detected"

    age = predict_age(model, face_image)
    return age

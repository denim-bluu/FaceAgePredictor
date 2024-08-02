from faceagepredictor.data.face_detection import FaceDetector
from faceagepredictor.prediction.predict import load_model, predict_from_image


def main():
    model_path = "path/to/saved/model.pth"
    image_path = "path/to/test/image.jpg"

    model = load_model(model_path)
    face_detector = FaceDetector()

    predicted_age = predict_from_image(image_path, model, face_detector)
    print(f"Predicted Age: {predicted_age:.2f} years")


if __name__ == "__main__":
    main()

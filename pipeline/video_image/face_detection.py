import cv2
import numpy as np


class FaceDetector:
    def __init__(
        self,
        cascade_path: str = "facedetection/video_image/haarcascade_frontalface_default.xml",
    ):
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        print("The number of faces found = ", len(faces))
        face_images = []
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face_images.append((face, (x, y, w, h)))
        return face_images

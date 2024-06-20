import numpy as np
from facenet_pytorch import MTCNN
import torch


class FaceDetector:
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.detector = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, frame: np.ndarray):
        boxes, _ = self.detector.detect(frame) # type: ignore
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                faces.append((face, (x1, y1, x2 - x1, y2 - y1)))
        print("The number of faces found = ", len(faces))
        return faces


# class FaceDetector:
#     def __init__(
#         self,
#         cascade_path: str = "facedetection/video_image/haarcascade_frontalface_default.xml",
#     ):
#         self.detector = cv2.CascadeClassifier(cascade_path)

#     def detect_faces(self, frame: np.ndarray):
#         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         faces = self.detector.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30),
#             flags=cv2.CASCADE_SCALE_IMAGE,
#         )
#         print("The number of faces found = ", len(faces))
#         face_images = []
#         for x, y, w, h in faces:
#             face = frame[y : y + h, x : x + w]
#             face_images.append((face, (x, y, w, h)))
#         return face_images

import torch
from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetector:
    def __init__(self, image_size=224, margin=0):
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.image_size = image_size

    def detect_face(self, image):
        """
        Detects a face in the given image and returns the cropped face.

        Args:
            image (PIL.Image): Input image

        Returns:
            PIL.Image: Cropped face image, or None if no face detected
        """
        try:
            boxes, _ = self.mtcnn.detect(image)  # type: ignore
            if boxes is not None:
                # Assuming we only care about the first detected face
                box = boxes[0]
                # Crop the face using the bounding box
                face = image.crop((box[0], box[1], box[2], box[3]))
                face = face.resize(
                    (self.image_size, self.image_size), Image.Resampling.LANCZOS
                )
                return face
        except Exception as e:
            print(f"Error in face detection: {e}")
        return None

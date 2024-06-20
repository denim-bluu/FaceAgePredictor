import logging
from datetime import datetime

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from pipeline.estimator.data_prep import get_transform
from pipeline.estimator.models import ModelFactory, predict_age
from pipeline.utils import get_device
from pipeline.video_image.face_detection import FaceDetector
from pipeline.video_image.video_to_image import extract_frames, save_frames_to_img


def preprocess_image(image: cv2.Mat, transform: transforms.Compose) -> torch.Tensor:
    image_pil = Image.fromarray(image).convert("RGB")
    image_transformed = transform(image_pil)
    image_tensor = torch.Tensor(image_transformed)
    return image_tensor.unsqueeze(0)


def process_video(
    config: DictConfig | ListConfig,
    model_name: str,
    weight_path: str,
    video_path: str,
    output_dir: str,
    frame_rate: int = 5,
    save_frames: bool = False,
) -> str:
    """
    Process a video to detect faces and predict their ages.

    Args:
        config (DictConfig | ListConfig): Configuration parameters.
        video_path (str): Path to the video file.
        frame_rate (int, optional): Frame rate for video processing. Defaults to 5.
        save_frames (bool, optional): Whether to save the frames or not. Defaults to False.

    Returns:
        str: Path to the processed video.
    """
    device = get_device()
    model = ModelFactory.create_model(model_name)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()
    transform = get_transform(config)

    face_detector = FaceDetector()
    frames = extract_frames(video_path, frame_rate)
    if save_frames:
        save_frames_to_img(frames, f"{output_dir}/frames")

    out = None
    predicted_ages = []

    logging.info(f"Processing {len(frames)} frames from video.")
    for frame in tqdm(frames, desc="Processing frames"):
        faces = face_detector.detect_faces(frame)
        for face, (x, y, w, h) in faces:
            image_tensor = preprocess_image(face, transform).to(device)
            if image_tensor is not None:
                age = predict_age(model, image_tensor)
                predicted_ages.append(age)
                cv2.putText(
                    frame,
                    f"Age: {age:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*"FMP4")  # type: ignore
            output_path = f"{output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
            out = cv2.VideoWriter(
                output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0])
            )

        out.write(frame)

    if out:
        out.release()

    if predicted_ages:
        estimated_age = np.mean(predicted_ages)
        logging.info(f"Estimated age for the video: {estimated_age:.2f}")
    else:
        logging.warning("No faces detected in the video.")

    logging.info(f"Processed video saved to: {output_path}")

    return output_path


def process_image(
    config: DictConfig | ListConfig,
    model_name: str,
    weight_path: str,
    output_dir: str,
    image_path: str,
) -> str | None:
    """
    Process an image to detect faces and predict their ages.

    Args:
        config (DictConfig | ListConfig): Configuration parameters.
        image_path (str): Path to the image file.

    Returns:
        str | None: Path to the processed image if faces are detected, None otherwise.
    """

    device = get_device()
    model = ModelFactory.create_model(model_name)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()
    transform = get_transform(config)

    face_detector = FaceDetector()

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)
    if image is None:
        logging.error(f"Could not read image: {image_path}")
        raise ValueError(f"Could not read image: {image_path}")

    faces = face_detector.detect_faces(image)
    if not faces:
        logging.warning("No faces detected in the image.")
        return None

    for face, (x, y, w, h) in faces:
        image_tensor = preprocess_image(face, transform).to(device)
        if image_tensor is not None:
            age = predict_age(model, image_tensor)
            cv2.putText(
                image,
                f"Age: {age:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            logging.info(f"Predicted age: {age:.2f}")

    output_path = f"{output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    logging.info(f"Processed image saved to: {output_path}")
    return output_path

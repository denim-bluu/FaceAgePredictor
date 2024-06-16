import logging
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from pipeline.config import Config
from pipeline.estimator.data_prep import get_transform
from pipeline.estimator.models import load_model, predict_age
from pipeline.utils import get_device, preprocess_image
from pipeline.video_image.face_detection import FaceDetector
from pipeline.video_image.video_to_image import extract_frames, save_frames_to_img


def process_video(
    video_path: str,
    model_name: str,
    model_path: str,
    output_dir: str,
    frame_rate: int = 5,
    save_frames: bool = False,
) -> str:
    device = get_device()
    model = load_model(model_name, model_path)
    transform = get_transform()

    face_detector = FaceDetector(cascade_path=Config.CASCADE_PATH)
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
    image_path: str, model_name: str, model_path: str, output_dir: str
) -> str | None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = get_device()
    model = load_model(model_name, model_path)
    transform = get_transform()

    face_detector = FaceDetector(cascade_path=Config.CASCADE_PATH)

    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read image: {image_path}")
        raise ValueError(f"Could not read image: {image_path}")

    faces = face_detector.detect_faces(image)
    if not faces:
        logging.warning("No faces detected in the image.")
        return

    for face, (x, y, w, h) in faces:
        image_tensor = preprocess_image(face, transform).to(device)
        if image_tensor is not None:
            age = predict_age(model, image_tensor)
            cv2.putText(
                image,
                f"Age: {age:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    output_path = f"{output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    logging.info(f"Processed image saved to: {output_path}")
    return output_path

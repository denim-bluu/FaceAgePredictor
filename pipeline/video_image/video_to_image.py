import cv2
import os
from typing import List


def extract_frames(video_path: str, frame_rate: int = 1) -> List[cv2.Mat]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_rate == 0:
                frames.append(frame)

            frame_count += 1
    except Exception as e:
        print(f"Error extracting frames: {e}")
    finally:
        cap.release()

    return frames


def save_frames_to_img(frames: List[cv2.Mat], output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(frames):
        output_dir = os.path.join(output_dir, f"frame_{i}.jpg")
        cv2.imwrite(output_dir, frame)

import logging
import os

import cv2
import ffmpeg
import psutil
import torch
from PIL import Image
from torchvision import transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_video(input_path: str, output_path: str):
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec="libx264")
        ffmpeg.run(stream)
    except ffmpeg.Error as e:
        print("Error:", e)


def log_resource_usage(prefix: str = "") -> None:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(
        f"{prefix} CPU: {psutil.cpu_percent()}% | RAM: {mem_info.rss / 1024 ** 2:.2f} MB"
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image: cv2.Mat, transform: transforms.Compose) -> torch.Tensor:
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    image_transformed = transform(image_pil)
    image_tensor = torch.Tensor(image_transformed)
    return image_tensor.unsqueeze(0)

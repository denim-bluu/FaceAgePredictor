import logging
import os

import ffmpeg
import GPUtil
import psutil
import torch
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_config(config_path: str = "config.yaml"):
    return OmegaConf.load(config_path)


def convert_video(input_path: str, output_path: str):
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec="libx264")
        ffmpeg.run(stream)
    except ffmpeg.Error as e:
        print("Error:", e)


# Function to log resource usage
def log_resource_usage(prefix=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    gpus = GPUtil.getGPUs()
    gpu_info = gpus[0] if gpus else None
    gpu_memory_used = gpu_info.memoryUsed if gpu_info else 0
    gpu_memory_total = gpu_info.memoryTotal if gpu_info else 0
    logging.info(
        f"{prefix} CPU: {psutil.cpu_percent()}% | RAM: {mem_info.rss / 1024 ** 2:.2f} MB | GPU: {gpu_memory_used}/{gpu_memory_total} MB"
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

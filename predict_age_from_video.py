import argparse

from pipeline.utils import get_config
from pipeline.processing import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict ages from a video input with face detection."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for prediction.",
    )
    parser.add_argument(
        "--weight_path", type=str, required=True, help="Path to the model weights."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed video.",
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video."
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=1,
        help="Frame rate to extract frames from the video.",
    )
    parser.add_argument(
        "--save_frames", action="store_true", help="Save preprocessed frames locally."
    )
    args = parser.parse_args()
    config = get_config()
    process_video(
        config,
        args.model_name,
        args.weight_path,
        args.video_path,
        args.output_dir,
        args.frame_rate,
        args.save_frames,
    )

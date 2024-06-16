import argparse

from pipeline.config import Config
from pipeline.processing import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict ages from a video input with face detection."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=Config.MODEL_NAME,
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=Config.WEIGHT_PATH,
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save the output video."
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

    process_video(
        args.video_path,
        args.model_name,
        args.model_path,
        args.output_dir,
        args.frame_rate,
        args.save_frames,
    )

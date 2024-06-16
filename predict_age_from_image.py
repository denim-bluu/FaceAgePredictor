import argparse

from pipeline.config import Config
from pipeline.processing import process_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict age from an image input with face detection."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
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
        "--output_dir", type=str, required=True, help="Path to save the output image."
    )
    args = parser.parse_args()

    process_image(args.image_path, args.model_name, args.model_path, args.output_dir)

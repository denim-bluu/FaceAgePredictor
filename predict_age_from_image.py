import argparse

from pipeline.utils import get_config
from pipeline.processing import process_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict age from an image input with face detection."
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
        help="Directory to save the processed image.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    args = parser.parse_args()
    config = get_config()
    process_image(
        config, args.model_name, args.weight_path, args.output_dir, args.image_path
    )

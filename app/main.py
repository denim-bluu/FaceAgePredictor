import os
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

from pipeline.utils import get_config
from pipeline.processing import process_image, process_video
from pipeline.utils import convert_video


def main():
    config = get_config("pipeline/config.yaml")
    st.title("Age Prediction from Image/Video")
    file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])
    model_name = st.text_input("Model Name", config.training.model_name or "")
    weight_path = st.text_input("Weight Path", config.training.weight_dir or "")
    output_dir = st.text_input("Output Directory", config.output.output_dir or "")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        if st.button("Predict"):
            if file.type == "video/mp4":
                video_path = process_video(
                    config,
                    model_name,
                    weight_path,
                    tfile.name,
                    output_dir,
                )
                converted_video_path = video_path.replace(".mp4", "_converted.mp4")
                convert_video(video_path, converted_video_path)
                st.video(converted_video_path)
            else:
                image = Image.open(tfile.name)
                image = np.array(image)
                st.image(image, caption="Uploaded Image.", use_column_width=True)
                image_path = process_image(
                    config,
                    model_name,
                    weight_path,
                    output_dir,
                    tfile.name,
                )
                if image_path is not None:
                    st.image(
                        image_path, caption="Processed Image.", use_column_width=True
                    )


if __name__ == "__main__":
    main()

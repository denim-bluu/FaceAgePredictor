import streamlit as st
from PIL import Image
import numpy as np
import tempfile

from pipeline.config import Config
from pipeline.utils import convert_video
from pipeline.processing import process_image, process_video


def main():
    st.title("Age Prediction from Image/Video")
    file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])
    model_name = st.text_input("Model Name", Config.MODEL_NAME)
    model_path = st.text_input("Model Path", Config.WEIGHT_PATH)
    output_dir = st.text_input("Output Directory", Config.OUTPUT_DIR)

    if file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        if st.button("Predict"):
            if file.type == "video/mp4":
                video_path = process_video(
                    tfile.name, model_name, model_path, output_dir
                )
                converted_video_path = video_path.replace(".mp4", "_converted.mp4")
                convert_video(video_path, converted_video_path)
                st.video(converted_video_path)
            else:
                image = Image.open(tfile.name)
                image = np.array(image)
                st.image(image, caption="Uploaded Image.", use_column_width=True)
                image_path = process_image(
                    tfile.name, model_name, model_path, output_dir
                )
                if image_path is not None:
                    st.image(
                        image_path, caption="Processed Image.", use_column_width=True
                    )


if __name__ == "__main__":
    main()

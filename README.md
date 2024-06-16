# Face Detection and Age Prediction System

## Overview

This project is a comprehensive system for detecting faces in images and videos and predicting the age of the detected faces using deep learning models. It includes functionalities for data preparation, model training and evaluation, and inference from images and videos. The project is structured to facilitate ease of use, scalability, and maintainability.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Streamlit App](#streamlit-app)

## Installation

Before you start, make sure you have `ffmpeg` installed on your system. This is required for video processing. You can install `ffmpeg` using the following command on a Linux machine:

```bash
sudo apt update
sudo apt install ffmpeg
```

This project uses Poetry for dependency management. If you haven't installed Poetry yet, you can do so following the guide [here](https://python-poetry.org/docs/).

Once Poetry is installed, you can install the project dependencies by running the following command:

```sh
poetry install
```

The Python version used in this project is 3.10.

## Dataset

The dataset used for training and validation of the model is the [Facial Age](https://www.kaggle.com/datasets/frabbisw/facial-age).

## Usage

### Training

To train the model, run the training pipeline script. This will load the data, initialise the model, train it, and evaluate its performance.

```sh
poetry run python pipeline/estimator/run_train_pipeline.py
```

#### Pretrained Weights

The pre-trained model weights are saved in the `pipeline/weights` directory. You can use these weights to skip the training process and directly use the model for inference.

### Inference

#### Predicting Age from an Image

To predict the age from an image, use the `predict_age_from_image.py` script. This script detects faces in the provided image and predicts the age for each detected face.

```sh
poetry run python predict_age_from_image.py --image_path path/to/image.jpg --output_dir path/to/output
```

#### Predicting Ages from a Video

To predict ages from a video, use the `predict_age_from_video.py` script. This script extracts frames from the video, detects faces in each frame, and predicts the age for each detected face.

```sh
poetry run python predict_age_from_video.py --video_path path/to/video.mp4 --output_dir path/to/output
```

## Configuration

The `config.py` file contains all the configuration settings for the project. You can adjust parameters such as data directories, batch size, image size, model paths, and more.

```python
class Config:
    DATA_DIR = "face_age"
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NUM_EPOCHS = 25
    ACCUMULATION_STEPS = 2
    LEARNING_RATE = 0.001
    WEIGHT_PATH = "pipeline/weights/trained_weights_alexnet_model.pth"
    MODEL_NAME = "AgeAlexNet"  # or "SmallCNN"
    ACCURACY_THRESHOLD = 5
    CASCADE_PATH = "pipeline/video_image/haarcascade_frontalface_default.xml"
    FRAME_OUTPUT_DIR = "preprocessed_frames"
```

## Streamlit App

This project includes a Streamlit app that provides a user-friendly interface for predicting ages from images and videos. To run the app, use the following command:

```sh
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser. You can upload an image or a video, and the app will display the detected faces and their predicted ages.

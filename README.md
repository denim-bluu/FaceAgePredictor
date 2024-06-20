# Face Detection and Age Prediction System

## Overview

This project is a simple system for detecting faces in images and videos and predicting the age of the detected faces using deep learning models. It includes functionalities for data preparation, model training and evaluation, and inference from images and videos.

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

The dataset used for training and validation of the model is the [UKT Face](https://www.kaggle.com/datasets/jangedoo/utkface-new).

## Usage

### Training

To train the model, run the training pipeline script. This will load the data, initialise the model, train it, and evaluate its performance.

```sh
poetry run python pipeline/estimator/run_train_pipeline.py
```

#### Pretrained Weights

The pre-trained model weights can be found in [this Google drive link](https://drive.google.com/file/d/1zqqeUoK3xUgK9HAFojzu_y7q8lYpJ2QV/view?usp=sharing).

### Inference

#### Predicting Age from an Image

To predict the age from an image, use the `predict_age_from_image.py` script. This script detects faces in the provided image and predicts the age for each detected face.

```sh
poetry run python predict_age_from_image.py --model_name AgeEfficientNet --weight_path path/to/weights --output_dir output_dir/ --image_path path/to/image.jpg --output_dir path/to/output
```

#### Predicting Ages from a Video

To predict ages from a video, use the `predict_age_from_video.py` script. This script extracts frames from the video, detects faces in each frame, and predicts the age for each detected face.

```sh
poetry run python predict_age_from_video.py --model_name AgeEfficientNet --weight_path path/to/weights --output_dir output_dir/ --video_path path/to/video.mp4
```

## Configuration

The `config.yaml` file contains all the configuration settings for the project. You can adjust parameters such as data directories, batch size, image size, model paths, and more.

```yaml
dataset:
  data_dir: "inputs/crop_part1"
  batch_size: 32
  image_size: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

training:
  num_epochs: 50
  accumulation_steps: 2
  learning_rate: 0.001
  weight_decay: 1e-4
  scheduler_factor: 0.1
  scheduler_patience: 5
  early_stopping_patience: 10
  accuracy_threshold: 5
  weight_dir: "pipeline/weights"
  model_name: "AgeEfficientNet"
  log_dir: "runs"
  checkpoint_dir: "checkpoint"

paths:
  cascade_path: "pipeline/video_image/haarcascade_frontalface_default.xml"

output:
  output_dir: "outputs"
```

## Streamlit App

This project includes a Streamlit app that provides a user-friendly interface for predicting ages from images and videos. To run the app, use the following command:

```sh
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser. You can upload an image or a video, and the app will display the detected faces and their predicted ages.

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
    OUTPUT_DIR = "outputs"

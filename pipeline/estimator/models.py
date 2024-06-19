import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b3


class ModelFactory:
    """
    A factory class for creating different types of models.

    Methods:
        create_model: Creates and returns a model based on the given model name.
    """
    @staticmethod
    def create_model(model_name: str) -> nn.Module:
        """
        Creates and returns a model based on the given model name.

        Args:
            model_name (str): The name of the model to create.

        Returns:
            nn.Module: The created model.

        Raises:
            ValueError: If the given model name is not supported.
        """
        if model_name == "SmallCNN":
            return SmallCNN()
        elif model_name == "AgeAlexNet":
            return AgeAlexNet()
        elif model_name == "AgeEfficientNet":
            return AgeEfficientNet()
        else:
            raise ValueError(f"Model {model_name} is not supported.")


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AgeAlexNet(nn.Module):
    def __init__(self):
        super(AgeAlexNet, self).__init__()
        self.alexnet = models.alexnet(weights="AlexNet_Weights.DEFAULT")
        self.alexnet.classifier[6] = nn.Linear(
            4096, 1
        )  # Output a single value for age prediction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alexnet(x)
        return x


class AgeEfficientNet(nn.Module):
    def __init__(self):
        super(AgeEfficientNet, self).__init__()
        self.efficientnet = efficientnet_b3(pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(
                self.efficientnet.classifier[1].in_features, 1
            ),  # Output a single value for age prediction
        )

    def forward(self, x):
        x = self.efficientnet(x)
        return x


def predict_age(model: torch.nn.Module, image_tensor: torch.Tensor) -> float:
    """
    Predicts the age based on the given image tensor using the given model.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        image_tensor (torch.Tensor): The image tensor to predict the age from.

    Returns:
        float: The predicted age.
    """
    with torch.no_grad():
        output = model(image_tensor)
    return output.item()

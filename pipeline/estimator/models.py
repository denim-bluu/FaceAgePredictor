import torch
import torch.nn as nn
from torchvision import models

from pipeline.utils import get_device


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


def load_model(model_name: str, model_path: str) -> nn.Module:
    device = get_device()

    if model_name == "SmallCNN":
        model = SmallCNN()
    elif model_name == "AgeAlexNet":
        model = AgeAlexNet()
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_age(model: torch.nn.Module, image_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        output = model(image_tensor)
    return output.item()

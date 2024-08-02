import torch
import torch.nn as nn
import torchvision.models as models

from .base_model import AgePredictor


class MobileNetAgePredictor(AgePredictor):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )
        self.freeze_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilenet(x)

    def predict(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            return self.forward(x).item()

    def freeze_layers(self) -> None:
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False
        for param in self.mobilenet.classifier.parameters():
            param.requires_grad = True

    def unfreeze_layers(self, num_layers: int) -> None:
        layers_to_unfreeze = list(reversed(list(self.mobilenet.features.children())))
        for layer in layers_to_unfreeze[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = True

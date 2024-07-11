import torch
import torch.nn as nn
import torchvision.models as models

from .base_model import AgePredictor


class ResNetAgePredictor(AgePredictor):
    """
    Age prediction model based on ResNet architecture.

    This model uses a pre-trained ResNet50 as a base and adds custom
    layers for age regression. It implements progressive unfreezing
    for transfer learning.
    """

    def __init__(self, weights="DEFAULT"):
        """
        Initializes the ResNetAgePredictor model.

        Args:
            pretrained (bool): Whether to use pre-trained weights for ResNet50
        """
        super().__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=weights)

        # Get the number of features from the last layer of ResNet
        num_ftrs = self.resnet.fc.in_features

        # Replace the last fully connected layer with a custom regressor
        self.resnet.fc = nn.Sequential( # type: ignore
            nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        ) 

        # Freeze the pre-trained layers
        self.freeze_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Age predictions for the input batch
        """
        return self.resnet(x).squeeze(1)

    def predict(self, x: torch.Tensor) -> float:
        """
        Makes a single age prediction.

        Args:
            x (torch.Tensor): Input tensor representing a single image

        Returns:
            float: Predicted age
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x).item()

    def freeze_layers(self) -> None:
        """
        Freezes all layers of ResNet and unfreezes the age regressor.
        """
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:  # Don't freeze the final custom layers
                param.requires_grad = False

    def unfreeze_layers(self, num_layers: int) -> None:
        """
        Unfreezes a specified number of ResNet layers for fine-tuning.

        Args:
            num_layers (int): Number of layers to unfreeze, starting from the top
        """
        layers_to_unfreeze = [
            self.resnet.layer4,
            self.resnet.layer3,
            self.resnet.layer2,
            self.resnet.layer1,
        ]
        for layer in layers_to_unfreeze[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = True

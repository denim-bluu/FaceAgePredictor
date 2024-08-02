from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
import torch.nn as nn


class AgePredictor(nn.Module, ABC):
    """
    Abstract base class for age prediction models.

    This class defines the interface for age prediction models, ensuring
    that all derived classes implement the necessary methods for forward
    pass, prediction, layer freezing/unfreezing, and parameter retrieval.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Age predictions for the input batch
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> float:
        """
        Makes a single age prediction.

        Args:
            x (torch.Tensor): Input tensor representing a single image

        Returns:
            float: Predicted age
        """
        pass

    @abstractmethod
    def freeze_layers(self) -> None:
        """
        Freezes the layers of the model.

        This method should set requires_grad to False for parameters
        that should not be updated during initial training phases.
        """
        pass

    @abstractmethod
    def unfreeze_layers(self, num_layers: int) -> None:
        """
        Unfreezes a specified number of layers for fine-tuning.

        Args:
            num_layers (int): Number of layers to unfreeze, starting from the top
        """
        pass

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """
        Retrieves the trainable parameters of the model.

        Returns:
            Iterator[nn.Parameter]: An iterator over the trainable parameters
        """
        return filter(lambda p: p.requires_grad, self.parameters())

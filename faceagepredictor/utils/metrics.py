import torch
import torch.nn as nn


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return nn.L1Loss()(predictions, targets).item()


def calculate_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(nn.MSELoss()(predictions, targets)).item()

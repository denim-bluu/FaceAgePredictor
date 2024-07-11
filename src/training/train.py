from typing import List

import torch
from src.models.base_model import AgePredictor
from torch import nn
from torch.utils.data import DataLoader
from src.utils.metrics import calculate_mae, calculate_rmse


def train_with_progressive_unfreezing(
    model: AgePredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    initial_lr: float,
    device: torch.device,
    unfreeze_schedule: List[int] = [5, 10, 15, 20],
) -> AgePredictor:
    """
    Trains the model using a progressive unfreezing strategy.

    This function implements a training loop with progressive unfreezing
    of layers and dynamic learning rate adjustment.

    Args:
        model (AgePredictor): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        num_epochs (int): Total number of epochs to train
        initial_lr (float): Initial learning rate
        device (torch.device): Device to use for training (CPU or GPU)
        unfreeze_schedule (List[int]): Epochs at which to unfreeze layers

    Returns:
        AgePredictor: The trained model
    """
    criterion = nn.L1Loss()

    for epoch in range(num_epochs):
        # Progressive unfreezing and learning rate adjustment
        if epoch in unfreeze_schedule:
            # Unfreeze layers based on schedule
            layers_to_unfreeze = unfreeze_schedule.index(epoch) + 1
            model.unfreeze_layers(layers_to_unfreeze)

            # Dynamically adjust learning rate based on number of layers unfrozen
            current_lr = initial_lr / (2**layers_to_unfreeze)
            optimizer = torch.optim.Adam(
                [{"params": model.get_trainable_params(), "lr": current_lr}]
            )
            print(
                f"Epoch {epoch+1}: Unfreezing {layers_to_unfreeze} layers, LR: {current_lr}"
            )
        elif epoch == 0:
            optimizer = torch.optim.Adam(model.get_trainable_params(), lr=initial_lr)

        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # Calculate and log metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae = calculate_mae(
            torch.tensor(val_predictions), torch.tensor(val_targets)
        )
        val_rmse = calculate_rmse(
            torch.tensor(val_predictions), torch.tensor(val_targets)
        )

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val MAE: {val_mae:.4f}, "
            f"Val RMSE: {val_rmse:.4f}"
        )

    return model

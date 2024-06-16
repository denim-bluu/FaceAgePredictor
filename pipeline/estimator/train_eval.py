import logging
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pipeline.config import Config
from pipeline.utils import get_device, log_resource_usage


def train_model(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: torch.nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = Config.NUM_EPOCHS,
    accumulation_steps: int = Config.ACCUMULATION_STEPS,
) -> torch.nn.Module:
    device = get_device()
    model.to(device)

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch}/{num_epochs - 1}")
        logging.info("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            optimizer.zero_grad()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.float().to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    loss = loss / accumulation_steps

                    if phase == "train":
                        loss.backward()
                        if (i + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                running_loss += loss.item() * inputs.size(0)
                correct_predictions += torch.sum(
                    torch.abs(outputs.squeeze() - labels) <= Config.ACCURACY_THRESHOLD
                ).item()
                total_predictions += labels.size(0)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_accuracy = correct_predictions / total_predictions * 100
            logging.info(
                f"{phase} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            )
            log_resource_usage(prefix=f"{phase} Epoch {epoch}")

    return model


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    accuracy_threshold: int = Config.ACCURACY_THRESHOLD,
) -> None:
    device = get_device()
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item() * inputs.size(0)

            preds = outputs.squeeze()
            correct_predictions += torch.sum(
                torch.abs(preds - labels) <= accuracy_threshold
            ).item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions * 100
    logging.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    log_resource_usage(prefix="Evaluation")

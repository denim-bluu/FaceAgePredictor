import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from pipeline.utils import get_device, log_resource_usage


class EarlyStopping:
    """
    A class used to implement early stopping in model training.

    Attributes:
        patience (int): How long to wait after last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement.
        counter (int): Number of epochs with no improvement after which training will be stopped.
        best_score (float): Best score so far.
        early_stop (bool): A flag that indicates whether early stopping should be used.
        val_loss_min (float): Minimum validation loss so far.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.verbose:
            logging.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss


class TrainingManager:
    """
    A class used to manage the training process of a model.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): The DataLoader for the training data.
        val_loader (DataLoader): The DataLoader for the validation data.
        criterion (torch.nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (optim.lr_scheduler.ReduceLROnPlateau): The learning rate scheduler.
        early_stopping (EarlyStopping): The early stopping handler.
        writer (SummaryWriter): The TensorBoard writer.
        num_epochs (int): The number of epochs to train for.
        accumulation_steps (int): The number of steps to accumulate gradients for before performing an update.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        early_stopping: "EarlyStopping",
        writer: SummaryWriter,
        num_epochs: int = 100,
        accumulation_steps: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.writer = writer
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps

    def train(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(self.num_epochs):
            self._train_epoch(epoch, device)
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                break

    def _train_epoch(self, epoch: int, device: torch.device) -> None:
        logging.info(f"Epoch {epoch}/{self.num_epochs - 1}")
        logging.info("-" * 10)

        # Training phase
        self.model.train()
        train_loss, train_corrects = 0.0, 0
        self.optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss = loss / self.accumulation_steps
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(
                torch.abs(outputs.squeeze() - labels) <= 5
            ).item()

        epoch_train_loss = train_loss / len(self.train_loader.dataset)  # type: ignore
        epoch_train_acc = train_corrects / len(self.train_loader.dataset) * 100  # type: ignore
        logging.info(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}%")
        self.writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        self.writer.add_scalar("Accuracy/train", epoch_train_acc, epoch)

        self._validate_epoch(epoch, device)

    def _validate_epoch(self, epoch: int, device: torch.device) -> None:
        self.model.eval()
        val_loss, val_corrects = 0.0, 0
        for inputs, labels in self.val_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(torch.abs(outputs.squeeze() - labels) <= 5).item()

        epoch_val_loss = val_loss / len(self.val_loader.dataset)  # type: ignore
        epoch_val_acc = val_corrects / len(self.val_loader.dataset) * 100  # type: ignore
        logging.info(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        self.writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        self.writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)

        self.scheduler.step(epoch_val_loss)
        self.early_stopping(epoch_val_loss, self.model)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    accuracy_threshold: int,
) -> None:
    """
    Evaluates a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (DataLoader): The DataLoader for the data to evaluate on.
        criterion (torch.nn.Module): The loss function.
        accuracy_threshold (int): The threshold for considering a prediction as correct.

    Returns:
        None
    """
    device = get_device()
    model.to(device)
    model.eval()

    total_loss, correct_predictions, total_predictions = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(
                torch.abs(outputs.squeeze() - labels) <= accuracy_threshold
            ).item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions * 100
    logging.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    log_resource_usage(prefix="Evaluation")

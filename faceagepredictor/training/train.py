import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from faceagepredictor.models.base_model import AgePredictor
from faceagepredictor.utils.metrics import calculate_mae, calculate_rmse


def train_with_progressive_unfreezing(
    model: AgePredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    initial_lr: float,
    device: torch.device,
    writer: SummaryWriter,
    unfreeze_schedule: list[int] = [5, 10, 15, 20],
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
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=initial_lr)

    best_val_loss = float("inf")
    layers_unfrozen = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Progressive unfreezing
        if epoch in unfreeze_schedule:
            layers_unfrozen += 1
            model.unfreeze_layers(layers_unfrozen)
            logger.info(f"Unfreezing layer group {layers_unfrozen}")
            new_lr = initial_lr * (0.1**layers_unfrozen)
            optimizer = torch.optim.Adam(model.get_trainable_params(), lr=new_lr)
            logger.info(f"Adjusted learning rate to {new_lr}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_predictions = []
        train_targets = []

        logger.info("Starting training phase...")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mae += calculate_mae(outputs, labels)
            train_predictions.extend(outputs.cpu().detach().numpy())
            train_targets.extend(labels.cpu().detach().numpy())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        train_rmse = calculate_rmse(
            torch.tensor(train_predictions), torch.tensor(train_targets)
        )

        # Log training metrics
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("MAE/train", avg_train_mae, epoch)
        writer.add_scalar("RMSE/train", train_rmse, epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_predictions = []
        val_targets = []

        logger.info("Starting validation phase...")
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            ):
                logger.info(
                    f"Processing validation batch {batch_idx+1}/{len(val_loader)}"
                )
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_mae += calculate_mae(outputs, labels)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        val_rmse = calculate_rmse(
            torch.tensor(val_predictions), torch.tensor(val_targets)
        )

        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        logger.info(
            f"Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}, Val RMSE: {val_rmse:.4f}"
        )

        # Log validation metrics
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("MAE/val", avg_val_mae, epoch)
        writer.add_scalar("RMSE/val", val_rmse, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "_temp.pth")
            logger.info("Saved new best model")

    logger.info("Training completed")
    return model

import torch

from src.data.data_prep import load_data
from src.models.resnet_model import ResNetAgePredictor
from src.training.train import train_with_progressive_unfreezing
from src.utils.load_config import load_config
from src.utils.metrics import calculate_mae, calculate_rmse


def main():
    config = load_config()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = config["paths"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    train_loader, val_loader, test_loader = load_data(data_dir, batch_size=batch_size)

    # Initialise model (choose between ResNet and MobileNet)
    model = ResNetAgePredictor().to(device)
    # model = MobileNetAgePredictor(pretrained=True).to(device)

    # Train the model
    trained_model = train_with_progressive_unfreezing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["num_epochs"],
        initial_lr=config["training"]["initial_lr"],
        device=device,
        unfreeze_schedule=config["training"]["unfreeze_schedule"],
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), config["paths"]["model_path"])

    # Evaluate on test set
    trained_model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = trained_model(inputs)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(targets.numpy())

    test_mae = calculate_mae(torch.tensor(test_predictions), torch.tensor(test_targets))
    test_rmse = calculate_rmse(
        torch.tensor(test_predictions), torch.tensor(test_targets)
    )

    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()

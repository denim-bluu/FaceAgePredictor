import torch
import torch.nn as nn

from estimator.data_prep import load_data
from estimator.models import AgeAlexNet, SmallCNN
from estimator.train_eval import evaluate_model, train_model


def main() -> None:
    # Load Data
    train_loader, val_loader = load_data("face_age")

    # Initialize Model
    model = SmallCNN()
    model = AgeAlexNet()

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Dataloaders Dictionary
    dataloaders = {"train": train_loader, "val": val_loader}

    # Train the Model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

    # Save the Model
    torch.save(model.state_dict(), "weights/age_prediction_model.pth")

    # Evaluate the Model
    evaluate_model(model, val_loader, criterion)


if __name__ == "__main__":
    main()

# run_train_pipeline.py
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

from pipeline.utils import get_config
from pipeline.estimator.data_prep import load_data
from pipeline.estimator.models import ModelFactory
from pipeline.estimator.train_eval import EarlyStopping, evaluate_model, TrainingManager


def main() -> None:
    config = get_config("pipeline/config.yaml")

    train_loader, val_loader = load_data(config)
    os.makedirs(config.training.log_dir, exist_ok=True)
    writer = SummaryWriter(config.training.log_dir)
    model = ModelFactory.create_model(config.training.model_name)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience, verbose=True
    )

    manager = TrainingManager(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        writer,
        num_epochs=config.training.num_epochs,
        accumulation_steps=config.training.accumulation_steps,
    )
    manager.train()
    writer.close()

    os.makedirs(config.training.weight_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        f"{config.training.weight_dir}/model_weights_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth",
    )
    evaluate_model(
        model,
        val_loader,
        criterion,
        accuracy_threshold=config.training.accuracy_threshold,
    )


if __name__ == "__main__":
    main()

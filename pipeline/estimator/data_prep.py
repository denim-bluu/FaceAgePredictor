# data_prep.py
import os
from typing import List, Tuple, Union

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_transform(config: DictConfig | ListConfig) -> transforms.Compose:
    """
    Returns a composed series of image transformations.

    Args:
        config (DictConfig | ListConfig): Configuration object containing dataset parameters.

    Returns:
        transforms.Compose: Composed image transformations.
    """
    return transforms.Compose(
        [
            transforms.Resize(tuple(config.dataset.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std),
        ]
    )


class AgePredictionDataset(Dataset):
    """
    A PyTorch Dataset for age prediction.

    Args:
        labels (List[Tuple[str, int]]): List of tuples containing image paths and corresponding labels.
        transform (transforms.Compose): Image transformations to apply.

    Returns:
        None
    """

    def __init__(
        self, labels: List[Tuple[str, int]], transform: transforms.Compose
    ) -> None:
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[torch.Tensor, Image.Image], torch.Tensor]:
        img_path, label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor


def load_data(config: DictConfig | ListConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the dataset and returns data loaders for training and validation sets.

    Args:
        config (DictConfig | ListConfig): Configuration object containing dataset parameters.

    Returns:
        Tuple[DataLoader, DataLoader]: Data loaders for training and validation sets.
    """
    file_paths = []
    labels = []
    for i in os.listdir(config.dataset.data_dir):
        split = i.split("_")
        labels.append(int(split[0]))
        file_paths.append(f"{config.dataset.data_dir}/{i}")

    df = pd.DataFrame({"x": file_paths, "label": labels})
    train_df, valid_df = train_test_split(
        df, train_size=0.7, shuffle=True, random_state=42
    )

    transform = get_transform(config)

    train_dataset = AgePredictionDataset(
        labels=train_df.values.tolist(), transform=transform
    )
    val_dataset = AgePredictionDataset(
        labels=valid_df.values.tolist(), transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.dataset.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.dataset.batch_size, shuffle=False
    )

    return train_loader, val_loader

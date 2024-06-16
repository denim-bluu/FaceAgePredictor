import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pipeline.config import Config


class AgePredictionDataset(Dataset):
    def __init__(
        self, labels: List[Tuple[str, int]], transform: transforms.Compose | None = None
    ):
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor | Image.Image, torch.Tensor]:
        img_path, label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ]
    )


def load_data(file_dir: str) -> Tuple[DataLoader, DataLoader]:
    file_paths = []
    labels = []

    for folder_name, _, filenames in os.walk(file_dir):
        if folder_name != file_dir:
            for f in filenames:
                file_paths.append(os.path.join(folder_name, f))
                labels.append(int(os.path.basename(folder_name)))

    df = pd.DataFrame({"x": file_paths, "label": labels})
    train_df, valid_df = train_test_split(
        df, train_size=0.7, shuffle=True, random_state=42
    )

    transform = get_transform()

    train_dataset = AgePredictionDataset(
        labels=train_df.values.tolist(), transform=transform
    )
    val_dataset = AgePredictionDataset(
        labels=valid_df.values.tolist(), transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

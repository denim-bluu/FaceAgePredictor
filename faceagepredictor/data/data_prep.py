import os

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import Compose


class FaceAgeDataset(Dataset):
    def __init__(self, root: str, transform: Compose | None = None):
        self.root = root
        self.transform = transform
        self.samples = [f for f in os.listdir(root) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, float]:
        img_name = self.samples[idx]
        img_path = os.path.join(self.root, img_name)

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        age = float(img_name.split("_")[0])
        if img is None:
            # Return a placeholder or skip this sample
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        return img, age


def get_transforms() -> Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_data(
    data_dir: str, batch_size: int = 32, test_size: float = 0.1, val_size: float = 0.1
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the dataset and returns data loaders for training, validation, and test sets.

    Args:
        data_dir (str): Directory containing the image files.
        batch_size (int): Batch size for the data loaders.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for training, validation, and test sets.
    """
    transform = get_transforms()

    full_dataset = FaceAgeDataset(root=data_dir, transform=transform)

    # First, split off the test set
    train_val_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=test_size, random_state=42
    )

    # Then split the remaining data into train and validation
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size / (1 - test_size), random_state=42
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader

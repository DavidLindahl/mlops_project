from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class TimmImageTransform:
    """Minimal PIL-to-tensor transform without requiring torchvision."""

    def __init__(self, image_size: int, mean: list[float], std: list[float]) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size))
        byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        x = byte_tensor.view(self.image_size, self.image_size, 3).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).div_(255.0)
        return (x - self.mean) / self.std


class MyDataset(Dataset):
    """My custom dataset for AI vs Human generated images."""

    def __init__(
        self,
        data_path: str | Path,
        limit: int | None = None,
        transform: Callable[[Image.Image], object] | None = None,
        target_transform: Callable[[object], object] | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to the dataset root containing `train.csv` and image files.
            limit: Optionally limit number of rows loaded from `train.csv` for quick experiments.
            transform: Optional transform applied to the image (e.g., resize/normalize to tensors).
            target_transform: Optional transform applied to the label.
        """
        self.data_path = Path(data_path)
        self.annotations = pd.read_csv(self.data_path / "train.csv", nrows=limit)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple:
        """Return a sample from the dataset.

        Args:
            index: Index of the sample to return.

        Returns:
            Tuple of (image, label) where image is a PIL Image.
        """
        img_name = self.annotations.iloc[index]["file_name"]
        image_path = self.data_path / img_name

        image = Image.open(image_path).convert("RGB")
        label = self.annotations.iloc[index]["label"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

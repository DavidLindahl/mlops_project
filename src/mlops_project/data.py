from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class NormalizeTransform:
    """Normalize a tensor image using ImageNet statistics."""

    def __init__(self, mean: Iterable[float], std: Iterable[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        x = byte_tensor.view(image.height, image.width, 3).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).div_(255.0)
        return (x - self.mean) / self.std


class ResizeNormalizeTransform:
    """Resize and normalize a PIL image for model input."""

    def __init__(self, image_size: int, mean: Iterable[float], std: Iterable[float]) -> None:
        self.image_size = image_size
        self.normalizer = NormalizeTransform(mean=mean, std=std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size))
        return self.normalizer(image)


def preprocess_dataset(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    image_size: int = 224,
    limit: int | None = None,
) -> Path:
    """Resize images ahead of training to avoid resizing at train time.

    This function writes resized images to `output_dir` and copies `train.csv`
    so the processed folder can be used as a dataset root.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(data_path / "train.csv", nrows=limit)
    for _, row in annotations.iterrows():
        img_name = row["file_name"]
        src_path = data_path / img_name
        dst_path = out_path / img_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(src_path).convert("RGB")
        image = image.resize((image_size, image_size))
        image.save(dst_path)

    annotations.to_csv(out_path / "train.csv", index=False)
    return out_path


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

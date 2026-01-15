import os
from collections.abc import Callable
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
import torch
from torch.utils.data import Dataset

KAGGLE_DATASET = "alessandrasala79/ai-vs-human-generated-dataset"


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

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def download_dataset(output_path: str | Path) -> None:
    """Download the AI vs Human Generated Images dataset from Kaggle.

    Args:
        output_path: Output directory for downloaded data.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset {KAGGLE_DATASET} to {output_dir}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=output_dir, unzip=True)
    print("Download complete!")


@hydra.main(config_path=str(Path(__file__).parent.parent.parent / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for data operations."""
    output_path = cfg.data.data_dir
    download_dataset(output_path)


if __name__ == "__main__":
    main()
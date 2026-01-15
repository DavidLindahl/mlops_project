from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset for AI vs Human generated images."""

    def __init__(self, data_path: str | Path, limit: int | None = None) -> None:
        """Initialize the dataset.

        Args:
            data_path: Path to the data directory containing train.csv and images.
            limit: Optional limit on number of samples to load.
        """
        self.data_path = Path(data_path)

        csv_path = self.data_path / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}. Did you run 'dvc pull'?")

        self.annotations = pd.read_csv(csv_path, nrows=limit)

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
        return image, label
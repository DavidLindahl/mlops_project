import os
from pathlib import Path
import pandas as pd
import typer
from PIL import Image
from torch.utils.data import Dataset

app = typer.Typer()


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, limit: int | None = None) -> None:
        self.data_path = data_path
        self.annotations = pd.read_csv(os.path.join(data_path, "train.csv"), nrows=limit)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        image_path = os.path.join(self.data_path, self.annotations.iloc[index]["file_name"])
        image = Image.open(image_path).convert("RGB")
        label = self.annotations.iloc[index]["label"]
        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


@app.command()
def preprocess(
    output_folder: str = typer.Argument(..., help="Output folder for processed data"),
    data_path: str = typer.Option("data/raw", "--data-path", "-d", help="Path to raw data"),
) -> None:
    """Preprocess the raw data and save it to the output folder."""
    print("Preprocessing data...")
    dataset = MyDataset(Path(data_path))
    dataset.preprocess(Path(output_folder))


if __name__ == "__main__":
    app()
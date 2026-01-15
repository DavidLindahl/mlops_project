import os
import zipfile
from pathlib import Path

import pandas as pd
import typer
from PIL import Image
from torch.utils.data import Dataset

app = typer.Typer()

KAGGLE_DATASET = "alessandrasala79/ai-vs-human-generated-dataset"


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
def download(
    output_path: str = typer.Option("data/raw", "--output", "-o", help="Output directory for downloaded data"),
) -> None:
    """Download the AI vs Human Generated Images dataset from Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset {KAGGLE_DATASET} to {output_dir}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=output_dir, unzip=True)
    print("Download complete!")





if __name__ == "__main__":
    app()
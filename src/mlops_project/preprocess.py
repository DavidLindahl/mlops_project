from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


def preprocess_dataset(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    image_size: int = 224,
    val_fraction: float = 0.2,
    seed: int = 42,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> Path:
    """Resize images ahead of training to avoid resizing at train time.

    This function writes resized images into train/val/test folders under
    `output_dir` and writes per-split CSV files.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = pd.read_csv(data_path / "train.csv")
    val_df = train_csv.sample(frac=val_fraction, random_state=seed)
    train_df = train_csv.loc[~train_csv.index.isin(val_df.index)]
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if train_limit is not None:
        sample_n = min(train_limit, len(train_df))
        train_df = train_df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    if val_limit is not None:
        sample_n = min(val_limit, len(val_df))
        val_df = val_df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    test_path = data_path / "test.csv"
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    if test_limit is not None and not test_df.empty:
        sample_n = min(test_limit, len(test_df))
        test_df = test_df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    split_map = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    for split, df in split_map.items():
        if df.empty:
            continue
        split_dir = out_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        if "file_name" in df.columns:
            img_col = "file_name"
        elif "id" in df.columns:
            img_col = "id"
        else:
            raise ValueError("Expected a 'file_name' or 'id' column in the CSV data.")

        new_names: list[str] = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Processing {split}"):
            img_path = getattr(row, img_col)
            src_path = data_path / img_path
            img_filename = Path(img_path).name
            dst_path = split_dir / img_filename

            image = Image.open(src_path).convert("RGB")
            image = image.resize((image_size, image_size))
            image.save(dst_path)
            new_names.append(img_filename)

        df_copy = df.copy()
        df_copy[img_col] = new_names

        # Save CSV in the processed root directory
        csv_name = f"{split}.csv"
        df_copy.to_csv(out_path / csv_name, index=False)
    return out_path


@hydra.main(config_path=str(Path(__file__).parent.parent.parent / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Preprocess raw images into a resized dataset for training."""
    preprocess_dataset(
        data_dir=cfg.data.data_dir,
        output_dir=cfg.data.processed_dir,
        image_size=cfg.data.image_size,
        val_fraction=cfg.train.val_fraction,
        seed=cfg.train.seed,
        train_limit=cfg.data.train_limit,
        val_limit=cfg.data.val_limit,
        test_limit=cfg.data.test_limit,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd
from mlops_project.preprocess import preprocess_dataset
from PIL import Image


def _write_raw_image(root: Path, name: str) -> None:
    Image.new("RGB", (20, 30)).save(root / name)


def test_preprocess_writes_splits_and_csvs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    # raw train images + train.csv
    rows = []
    for i in range(5):
        fn = f"img{i}.jpg"
        _write_raw_image(raw, fn)
        rows.append({"file_name": fn, "label": i % 2})
    pd.DataFrame(rows).to_csv(raw / "train.csv", index=False)

    # raw test images + test.csv (id column)
    test_rows = []
    for i in range(3):
        fn = f"test{i}.jpg"
        _write_raw_image(raw, fn)
        test_rows.append({"id": fn})
    pd.DataFrame(test_rows).to_csv(raw / "test.csv", index=False)

    out = tmp_path / "processed"
    preprocess_dataset(
        data_dir=raw,
        output_dir=out,
        image_size=8,
        val_fraction=0.2,
        seed=42,
        train_limit=2,
        val_limit=2,
        test_limit=2,
    )

    # csvs exist
    assert (out / "train.csv").exists()
    assert (out / "val.csv").exists()
    assert (out / "test.csv").exists()

    # dirs exist
    assert (out / "train").exists()
    assert (out / "val").exists()
    assert (out / "test").exists()

    # images resized
    some_train_img = next((out / "train").iterdir())
    img = Image.open(some_train_img)
    assert img.size == (8, 8)

from pathlib import Path

import pandas as pd
from mlops_project.data import MyDataset
from PIL import Image
from torch.utils.data import Dataset


def test_my_dataset(tmp_path: Path):
    # Create minimal dataset structure
    img_dir = tmp_path
    img_path = img_dir / "img0.jpg"
    Image.new("RGB", (8, 8)).save(img_path)

    # Create train.csv expected by MyDataset
    df = pd.DataFrame([{"file_name": "img0.jpg", "label": 0}])
    df.to_csv(tmp_path / "train.csv", index=False)

    dataset = MyDataset(tmp_path)
    assert isinstance(dataset, Dataset)
    x, y = dataset[0]
    assert y in (0, 1)  # basic sanity

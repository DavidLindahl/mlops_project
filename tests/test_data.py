from torch.utils.data import Dataset

from mlops_project.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/train_data")
    assert isinstance(dataset, Dataset)

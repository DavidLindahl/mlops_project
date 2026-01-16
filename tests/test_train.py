import random

import numpy as np
import torch
from mlops_project.train import _seed_everything


def test_seed_everything_deterministic() -> None:
    _seed_everything(123)
    a1 = random.random()
    b1 = np.random.rand(3)
    c1 = torch.rand(3)

    _seed_everything(123)
    a2 = random.random()
    b2 = np.random.rand(3)
    c2 = torch.rand(3)

    assert a1 == a2
    assert np.allclose(b1, b2)
    assert torch.allclose(c1, c2)

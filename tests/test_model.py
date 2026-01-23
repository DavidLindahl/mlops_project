from __future__ import annotations

import mlops_project.model as model_mod
import torch


def test_model_forward_shape_offline() -> None:
    m = model_mod.Model(pretrained=False)
    x = torch.rand(2, 3, 8, 8)
    y = m(x)
    assert y.shape == (2, 2)

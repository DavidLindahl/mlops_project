from __future__ import annotations

import timm
import torch
from timm.data import resolve_model_data_config
from torch import nn


class Model(nn.Module):
    """A 2-class image classifier built with `timm`.

    This wraps a `timm` backbone and exposes a simple `nn.Module` that outputs
    logits for classification (2 classes by default).
    """

    def __init__(self, *, pretrained: bool = False) -> None:
        """Initialize the model.

        Args:
            pretrained: Whether to load pretrained ImageNet weights. Defaults to
                False to avoid implicit downloads in offline/CI environments.
        """
        super().__init__()
        self.model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
        self.num_classes = 2
        self.model: nn.Module = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
        )
        self.data_config = resolve_model_data_config(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from an input batch.

        Args:
            x: Input tensor of shape (N, 3, H, W). H and W should match the
                expected model input size.

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        return self.model(x)


if __name__ == "__main__":
    model = Model()
    input_size = model.data_config["input_size"]
    x = torch.rand(2, input_size[0], input_size[1], input_size[2])
    print(f"Output shape of model: {model(x).shape}")

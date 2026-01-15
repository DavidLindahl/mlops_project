from __future__ import annotations

import timm
import torch
from torch import nn


class Model(nn.Module):
    """Vision Transformer (ViT) 2-class classifier built with `timm`.

    This wraps `timm`'s `vit_base_patch16_224` and exposes a simple `nn.Module`
    that outputs logits for binary classification (2 classes by default).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        """Initialize the model.

        Args:
            num_classes: Number of output classes for classification.
            pretrained: Whether to load pretrained ImageNet weights. Defaults to
                False to avoid implicit downloads in offline/CI environments.
        """
        super().__init__()
        self.model: nn.Module = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from an input batch.

        Args:
            x: Input tensor of shape (N, 3, H, W). For this ViT variant, H=W=224
                is expected (unless you adapt preprocessing/resizing).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        return self.model(x)


if __name__ == "__main__":
    model = Model()
    x = torch.rand(2, 3, 224, 224)
    print(f"Output shape of model: {model(x).shape}")

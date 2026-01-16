from __future__ import annotations

import torch
from torch import nn


class Model(nn.Module):
    """A 2-class image classifier built with `timm`.

    This wraps a `timm` backbone and exposes a simple `nn.Module` that outputs
    logits for classification (2 classes by default).
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        num_classes: int = 2,
        pretrained: bool = False,
    ) -> None:
        """Initialize the model.

        Args:
            model_name: Name of the `timm` model to instantiate.
            num_classes: Number of output classes for classification.
            pretrained: Whether to load pretrained ImageNet weights. Defaults to
                False to avoid implicit downloads in offline/CI environments.
        """
        # local imports
        # should stay in init as otherwise
        # will be loaded during tests
        # slowing everything down
        import timm  # local import
        from timm.data import resolve_model_data_config  # local import

        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
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

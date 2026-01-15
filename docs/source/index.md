## Documentation

This project provides a simple image classification pipeline for the AI vs Human Generated Images dataset.
The core components are data loading, a `timm` model wrapper, and a Hydra-based training entrypoint.

### Data loading

Data loading is handled by `MyDataset` in `mlops_project.data`. It reads annotations from `train.csv`
in `data/raw` and returns `(image, label)` pairs, where `image` is a PIL RGB image. The dataset supports
optional `transform` and `target_transform` callables.

For training, a minimal image transform is provided as `TimmImageTransform`. It resizes images to the
input size expected by the chosen `timm` model and applies ImageNet-style normalization using the model's
data configuration.

### Model

The model is defined in `mlops_project.model` as a thin wrapper around `timm.create_model`. The default
backbone is `tf_efficientnetv2_s.in21k_ft_in1k`, configured for binary classification (`num_classes=2`).
The wrapper exposes `data_config` so training can automatically set the correct input size and normalization.

### Training

Training is implemented in `mlops_project.train` and is driven by Hydra config files under `configs/`.
Key settings live in:

- `configs/model.yaml` for model name, class count, and pretrained weights
- `configs/train.yaml` for optimizer/scheduler settings and training parameters
- `configs/data.yaml` for data paths and loader settings

Running `uv run train` uses the default configuration. Overrides can be passed on the command line, for example:

```
uv run train train.batch_size=64 train.lr=1e-4 model.pretrained=false
```

Each run writes outputs to the Hydra run directory, including a `metrics.csv` file and checkpoints under
`checkpoints/` (best and last).

### Evaluation

Evaluation is implemented in `mlops_project.evaluate`. It loads the Hydra config from the run directory to
recreate the preprocessing pipeline and model settings, then evaluates the best checkpoint.

By default, `uv run eval` evaluates the latest run under `reports/runs`. You can also evaluate a run you
copied into the `models/` folder:

```
uv run eval
uv run eval my_copied_run
```

Evaluation results are saved to `eval_metrics.csv` in the run directory.

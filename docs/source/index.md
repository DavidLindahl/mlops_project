## Documentation

This project provides a simple image classification pipeline for the AI vs Human Generated Images dataset.
The core components are data loading, a `timm` model wrapper, and a Hydra-based training entrypoint.

### Data loading

Data loading is handled by `MyDataset` in `mlops_project.data`. It reads annotations from `train.csv`
in `data/raw` and returns `(image, label)` pairs, where `image` is a PIL RGB image. The dataset supports
optional `transform` and `target_transform` callables.

For training, the dataset can be preprocessed ahead of time using `preprocess_dataset` in `mlops_project.data`.
This resizes images once and stores them under a processed folder, so training does not pay the resize cost.
At train time, `NormalizeTransform` applies ImageNet-style normalization.

To generate the processed dataset, run:

```
uv run preprocess
```

### Model

The model is defined in `mlops_project.model` as a thin wrapper around `timm.create_model`. The backbone
is fixed to `tf_efficientnetv2_s.in21k_ft_in1k`, configured for binary classification (`num_classes=2`).
The wrapper exposes `data_config` to provide ImageNet normalization statistics.

### Training

Training is implemented in `mlops_project.train` and is driven by Hydra config files under `configs/`.
Key settings live in:

- `configs/train.yaml` for optimizer/scheduler settings and training parameters
- `configs/data.yaml` for data paths and loader settings

Training uses the processed dataset only. Running `uv run train` uses the default configuration. Overrides can be
passed on the command line, for example:

```
uv run train train.batch_size=64 train.lr=1e-4 train.pretrained=false
```

Each run writes outputs to the Hydra run directory, including a `metrics.csv` file and checkpoints under
`checkpoints/` (best and last).

### Evaluation

Evaluation is implemented in `mlops_project.evaluate`. It loads the Hydra config from the run directory to
recreate the preprocessing pipeline and model settings, then evaluates the best checkpoint.

Evaluation only operates on runs copied into the `models/` folder:

```
uv run eval my_copied_run
```

If no model name is provided, the latest folder under `models/` is used.

Evaluation results are saved to `eval_metrics.csv` in the run directory.

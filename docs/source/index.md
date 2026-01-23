## Documentation

This project provides a simple image classification pipeline for the AI vs Human Generated Images dataset.
The core components are data loading, a `timm` model wrapper, and a Hydra-based training entrypoint.

### Quick Start

- **Local Development**: See sections below for local training and evaluation
- **GCP Vertex AI**: See [GCP Vertex AI Setup Guide](../GCP_VERTEX_AI_SETUP.md) for cloud-based training on Google Cloud Platform

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

### GCP Vertex AI Training

This project supports training on Google Cloud Platform using Vertex AI Custom Jobs. This provides:

- **Scalable GPU resources** - T4, V100, or A100 GPUs on demand
- **Reproducible environments** - Docker containers ensure consistency
- **Cost-effective** - Pay only for compute time used
- **Cloud-native data** - Direct access to GCS buckets via `/gcs/` mount

**Quick workflow:**

1. Build and push Docker images to Artifact Registry
2. Upload data to GCS bucket
3. Submit preprocessing job (optional, can run locally)
4. Submit training job with GPU
5. Monitor and download results

**Documentation:**
- **[GCP Workflow Guide](../GCP_WORKFLOW.md)** - Quick reference for daily operations
- **[GCP Vertex AI Setup](../GCP_VERTEX_AI_SETUP.md)** - Complete setup instructions

**Key files:**
- `dockerfiles/vertex_train.dockerfile` - Training container with CUDA support
- `dockerfiles/vertex_preprocess.dockerfile` - Lightweight preprocessing container
- `configs/vertex_train_config.yaml` - Vertex AI training job configuration
- `configs/vertex_preprocess_config.yaml` - Vertex AI preprocessing job configuration

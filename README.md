# mlops_project

Image classification pipeline for detecting AI-generated vs human-generated images. The project uses a timm
backbone, Hydra-based configuration, and a FastAPI inference service. It also includes a Vertex AI workflow
for scalable training and preprocessing on GCP.

## Quick start

```bash
uv sync --dev
uv run dvc pull
```

Preprocess data and train with the default configuration:

```bash
uv run preprocess
uv run train
```

Run the API locally:

```bash
MODEL_PATH=/path/to/checkpoint.ckpt uv run python -m mlops_project.api
```

Send images to the API (e.g. the test folder):

```bash
uv run inference predict /path/to/images --api-url http://localhost:8000
```

## Project structure

```txt
├── .github/                  # GitHub workflows and automation
│   ├── dependabot.yaml
│   └── workflows/
│       ├── linting.yaml
│       └── tests.yaml
├── configs/                  # Hydra configuration files
│   ├── config.yaml
│   ├── data.yaml
│   ├── logging.yaml
│   ├── model.yaml
│   ├── train.yaml
│   ├── vertex_preprocess_config.yaml
│   └── vertex_train_config.yaml
├── data/                     # DVC-tracked datasets and CSV metadata
│   ├── train.csv
│   ├── test.csv
│   └── *.dvc
├── dockerfiles/              # Vertex AI Docker images
│   ├── vertex_preprocess.dockerfile
│   └── vertex_train.dockerfile
├── docs/                     # MkDocs documentation and guides
│   ├── API_DEPLOYMENT.md
│   ├── GCP_VERTEX_AI_SETUP.md
│   ├── GCP_WORKFLOW.md
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── models/                   # Trained runs copied for inference
├── notebooks/                # Exploratory notebooks
├── reports/                  # Evaluation reports
│   ├── README.md
│   └── report.py
├── src/                      # Source code
│   └── mlops_project/
│       ├── api.py
│       ├── data.py
│       ├── inference.py
│       ├── model.py
│       ├── preprocess.py
│       └── train.py
├── tests/                    # Test suite
├── pyproject.toml            # Project metadata and scripts
└── tasks.py                  # Invoke tasks
```

## Docs

Serve documentation locally:

```bash
uv run mkdocs serve --config-file docs/mkdocs.yaml
```

## Quality checks

```bash
uv run ruff check . --fix
uv run ruff format .
uv run mypy .
uv run pytest tests/
```

## Credits

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a
[cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with MLOps.

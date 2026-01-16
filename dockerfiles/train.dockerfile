FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project

# Important: Copy .dvc and .git folders so 'dvc pull' knows where to get data
COPY .dvc .dvc
COPY .dvcignore .dvcignore

ENV DVC_NO_SCM=true


COPY data data/
COPY configs configs/
COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/mlops_project/train.py"]

# Default command (can be overridden by Vertex AI)
# This chains the pull AND the training
CMD ["uv run dvc pull && uv run train"]
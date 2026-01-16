FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
#docker build -f dockerfiles/train.dockerfile . -t train:latest
#docker run -v "$PWD/data:/data" train:testlimit data.data_dir=/data

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
COPY configs configs/
COPY .dvc .dvc
COPY .dvcignore .dvcignore

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "train"]

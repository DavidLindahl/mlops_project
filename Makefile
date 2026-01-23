# GCP Vertex AI Automation
# Usage: make <target> [VARIABLE=value]

PROJECT_ID ?= mlops-485010
REGION ?= europe-west1
REPO_NAME ?= mlops-training
BUCKET ?= gs://mlops-training-stdne
TAG ?= v1

# Docker image URIs
TRAIN_IMAGE = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/mlops-trainer:$(TAG)
PREPROCESS_IMAGE = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/mlops-preprocessor:$(TAG)

.PHONY: help build-train build-preprocess push-train push-preprocess submit-train submit-preprocess promote-model list-runs

help:
	@echo "GCP Vertex AI Commands:"
	@echo "  make build-train [TAG=v1]          - Build training Docker image"
	@echo "  make build-preprocess [TAG=v1]      - Build preprocessing Docker image"
	@echo "  make push-train [TAG=v1]            - Build and push training image"
	@echo "  make push-preprocess [TAG=v1]       - Build and push preprocessing image"
	@echo "  make submit-train                   - Submit training job"
	@echo "  make submit-preprocess               - Submit preprocessing job"
	@echo "  make promote-model RUN=<path> VERSION=<v1>  - Promote model to production"
	@echo "  make list-runs                       - List available training runs"

build-train:
	docker build --platform linux/amd64 \
		-f dockerfiles/vertex_train.dockerfile \
		-t $(TRAIN_IMAGE) .

build-preprocess:
	docker build --platform linux/amd64 \
		-f dockerfiles/vertex_preprocess.dockerfile \
		-t $(PREPROCESS_IMAGE) .

push-train: build-train
	docker push $(TRAIN_IMAGE)
	@echo "✅ Image pushed: $(TRAIN_IMAGE)"

push-preprocess: build-preprocess
	docker push $(PREPROCESS_IMAGE)
	@echo "✅ Image pushed: $(PREPROCESS_IMAGE)"

submit-train:
	gcloud ai custom-jobs create \
		--region=$(REGION) \
		--display-name="mlops_training_$$(date +%Y%m%d_%H%M%S)" \
		--config=configs/vertex_train_config.yaml \
		--project=$(PROJECT_ID)

submit-preprocess:
	gcloud ai custom-jobs create \
		--region=$(REGION) \
		--display-name="mlops_preprocess_$$(date +%Y%m%d_%H%M%S)" \
		--config=configs/vertex_preprocess_config.yaml \
		--project=$(PROJECT_ID)

promote-model:
	@if [ -z "$(RUN)" ]; then \
		echo "❌ Error: RUN is required. Usage: make promote-model RUN=2026-01-23/10-47-37 VERSION=v1"; \
		exit 1; \
	fi
	@VERSION=$${VERSION:-$$(echo $(RUN) | cut -d'/' -f1 | tr -d '-')}; \
	echo "📦 Promoting model from $(RUN) to version $$VERSION..."; \
	gsutil cp $(BUCKET)/runs/$(RUN)/checkpoints/best_model.pt $(BUCKET)/models/$$VERSION/model.pt && \
	gsutil cp $(BUCKET)/models/$$VERSION/model.pt $(BUCKET)/models/latest/model.pt && \
	echo "✅ Model promoted to models/$$VERSION/ and models/latest/"

list-runs:
	@echo "📋 Available training runs:"
	@gsutil ls $(BUCKET)/runs/ | tail -10 || echo "No runs found"

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
	@echo "  make promote-model [RUN=<path>] [VERSION=<v1>]  - Promote model (auto-finds latest if RUN not specified)"
	@echo "  make list-runs [DATE=2026-01-23]     - List available training runs (optionally filter by date)"

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
		echo "🔍 Finding latest run..."; \
		LATEST_DATE=$$(gsutil ls $(BUCKET)/runs/ | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}/$$' | sort -r | head -1 | xargs basename); \
		if [ -z "$$LATEST_DATE" ]; then \
			echo "❌ No runs found in $(BUCKET)/runs/"; \
			exit 1; \
		fi; \
		LATEST_RUN=$$(gsutil ls $(BUCKET)/runs/$$LATEST_DATE/ | grep -v '^$$' | sort -r | head -1 | xargs basename); \
		if [ -z "$$LATEST_RUN" ]; then \
			echo "❌ No runs found for date $$LATEST_DATE"; \
			exit 1; \
		fi; \
		SELECTED_RUN="$$LATEST_DATE/$$LATEST_RUN"; \
		echo "✅ Found latest run: $$SELECTED_RUN"; \
	else \
		SELECTED_RUN="$(RUN)"; \
	fi; \
	VERSION=$${VERSION:-$$(echo $$SELECTED_RUN | cut -d'/' -f1 | tr -d '-')}; \
	echo "📦 Promoting model from $$SELECTED_RUN to version $$VERSION..."; \
	gsutil cp $(BUCKET)/runs/$$SELECTED_RUN/checkpoints/best_model.pt $(BUCKET)/models/$$VERSION/model.pt && \
	gsutil cp $(BUCKET)/models/$$VERSION/model.pt $(BUCKET)/models/latest/model.pt && \
	echo "✅ Model promoted to models/$$VERSION/ and models/latest/"

list-runs:
	@if [ -z "$(DATE)" ]; then \
		echo "📋 Available training runs:"; \
		echo ""; \
		for date_dir in $$(gsutil ls $(BUCKET)/runs/ | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}/$$'); do \
			date=$$(basename $$date_dir); \
			echo "📅 $$date:"; \
			gsutil ls $$date_dir 2>/dev/null | grep -v '^$$' | while read run; do \
				run_name=$$(basename $$run); \
				echo "   └─ $$run_name"; \
			done || echo "   (no runs)"; \
			echo ""; \
		done || echo "No runs found"; \
	else \
		echo "📋 Training runs for $(DATE):"; \
		gsutil ls $(BUCKET)/runs/$(DATE)/ 2>/dev/null | grep -v '^$$' | while read run; do \
			run_name=$$(basename $$run); \
			echo "   └─ $$run_name"; \
		done || 			echo "   (no runs found for $(DATE))"; \
		fi

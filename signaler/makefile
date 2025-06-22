# Makefile for Trading Signal System (pyproject.toml version)

.PHONY: help setup install test lint format clean deploy backfill train predict api docker-build docker-up docker-down

# Python version from .python-version file
PYTHON_VERSION := 3.11.6

# Default target
help:
	@echo "Trading Signal System - Available commands:"
	@echo "  make setup          - Set up development environment with pyenv"
	@echo "  make install        - Install dependencies"
	@echo "  make install-dev    - Install with all development dependencies"
	@echo "  make install-prod   - Install minimal production dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo "  make deploy        - Deploy to GCP"
	@echo "  make backfill      - Run data backfill"
	@echo "  make train         - Train model"
	@echo "  make predict       - Generate predictions"
	@echo "  make api           - Run API server locally"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"

# Environment setup with pyenv
setup:
	@echo "Setting up development environment..."
	@if ! command -v pyenv >/dev/null 2>&1; then \
		echo "❌ pyenv not found. Please install pyenv first:"; \
		echo "  macOS: brew install pyenv"; \
		echo "  Linux: curl https://pyenv.run | bash"; \
		exit 1; \
	fi
	pyenv install -s $(PYTHON_VERSION)
	pyenv local $(PYTHON_VERSION)
	echo "$(PYTHON_VERSION)" > .python-version
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -e ".[dev,test,lint]"
	cp .env.example .env
	@echo "✅ Setup complete! Don't forget to:"
	@echo "1. Activate venv: source venv/bin/activate"
	@echo "2. Edit .env with your configuration"

# Install dependencies
install:
	pip install --upgrade pip setuptools wheel
	pip install -e .

# Install with development dependencies
install-dev:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev,test,lint,docs]"

# Install for production (minimal dependencies)
install-prod:
	pip install --upgrade pip setuptools wheel
	pip install .

# Install specific dependency groups
install-test:
	pip install -e ".[test]"

install-lint:
	pip install -e ".[lint]"

install-docs:
	pip install -e ".[docs]"

# Testing
test: install-test
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit: install-test
	pytest tests/ -v -m "not integration" --cov=src

test-integration: install-test
	pytest tests/ -v -m integration

test-fast: install-test
	pytest tests/ -v -x --ff

# Code quality
lint: install-lint
	flake8 src/ tests/
	mypy src/

format: install-dev
	black src/ tests/
	isort src/ tests/

format-check: install-dev
	black --check src/ tests/
	isort --check-only src/ tests/

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf .mypy_cache/

# Build and distribution
build: clean
	pip install build
	python -m build

build-wheel: clean
	pip install build
	python -m build --wheel

build-sdist: clean
	pip install build
	python -m build --sdist

# GCP Deployment
deploy:
	@echo "Deploying to GCP..."
	./scripts/deploy.sh --project-id $(GCP_PROJECT_ID) --region $(GCP_REGION)

deploy-terraform:
	cd terraform && terraform init
	cd terraform && terraform plan -var="project_id=$(GCP_PROJECT_ID)"
	cd terraform && terraform apply -var="project_id=$(GCP_PROJECT_ID)" -auto-approve

# Data operations
backfill:
	trading-backfill \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-types all

backfill-ohlcv:
	trading-backfill \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-types ohlcv

daily-update:
	trading-daily-ingestion --date $(DATE)

# Model operations
train:
	trading-train \
		--end-date $(END_DATE) \
		--lookback-months 24

train-validate:
	trading-train --validate-only

predict:
	trading-predict \
		--model-version latest \
		--prediction-date $(DATE)

# API operations
api: install-dev
	uvicorn src.api.prediction_service:app --reload --host 0.0.0.0 --port 8080

api-prod: install-prod
	gunicorn src.api.prediction_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080

# Docker operations
docker-build:
	docker build -f docker/Dockerfile.base -t trading-system/base:latest . \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION)
	docker build -f docker/Dockerfile.ingestion -t trading-system/ingestion:latest . \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION)
	docker build -f docker/Dockerfile.api -t trading-system/api:latest . \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION)

docker-up:
	GCP_PROJECT_ID=$(GCP_PROJECT_ID) BQ_DATASET=$(BQ_DATASET) docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

# BigQuery operations
bq-init:
	python -c "from src.utils.bigquery import BigQueryClient; BigQueryClient().create_dataset_if_not_exists(); BigQueryClient().create_tables()"

bq-validate:
	python -c "from src.data_ingestion.data_validator import DataValidator; print('Data validation completed')"

# MLflow operations
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Monitoring
monitor-start:
	docker-compose up -d prometheus grafana

monitor-stop:
	docker-compose stop prometheus grafana

# Development helpers
notebook: install-dev
	jupyter lab --notebook-dir=notebooks/

docs: install-docs
	mkdocs serve

docs-build: install-docs
	mkdocs build

# CI/CD helpers
ci-test: install-test
	pytest tests/ -v --cov=src --cov-report=xml

ci-lint: install-lint
	flake8 src/ tests/ --exit-zero
	mypy src/ --ignore-missing-imports

ci-build: build

# Environment check
check-env:
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Virtual environment: $(VIRTUAL_ENV)"
	@echo "Pyenv version: $(shell pyenv version 2>/dev/null || echo 'Not using pyenv')"

# Variables with defaults
GCP_PROJECT_ID ?= your-project-id
GCP_REGION ?= us-central1
START_DATE ?= 2022-01-01
END_DATE ?= $(shell date +%Y-%m-%d)
DATE ?= $(shell date +%Y-%m-%d)
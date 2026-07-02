.PHONY: help env install dev test lint format typecheck run serve up down migrate clean

CONDA_ENV ?= AIEmployee
PY ?= python

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env:                 ## Create the conda environment
	conda env create -f environment.yml

install:             ## Install the package + runtime deps
	pip install -r requirements.txt && pip install -e .

dev:                 ## Install dev dependencies + editable package
	pip install -r requirements-dev.txt && pip install -e .

test:                ## Run the test suite (offline, mock provider)
	LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock MEMORY_BACKEND=memory pytest

lint:                ## Lint with ruff
	ruff check src tests

format:              ## Format with black + ruff
	black src tests && ruff check --fix src tests

typecheck:           ## Static type check with mypy
	mypy src

run:                 ## Run a goal end-to-end (make run GOAL="...")
	$(PY) -m ai_employee run "$(GOAL)"

serve:               ## Start the API server locally
	$(PY) -m ai_employee serve

up:                  ## Start the full stack with Docker Compose
	docker compose up --build

down:                ## Stop the Docker stack
	docker compose down

migrate:             ## Apply database migrations
	alembic upgrade head

clean:               ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

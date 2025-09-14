# Crown Template Makefile
.PHONY: help install dev-install format lint type-check test clean docker-up docker-down gpu-up gpu-down

# Default target
help:
	@echo "Crown Template Python Service"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev-install  Install dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format       Format code with ruff"
	@echo "  make lint         Lint code with ruff"
	@echo "  make type-check   Type check with mypy"
	@echo "  make test         Run tests with pytest"
	@echo "  make clean        Clean cache files"
	@echo ""
	@echo "Services:"
	@echo "  make docker-up    Start local services (postgres, redis, minio)"
	@echo "  make docker-down  Stop local services"
	@echo ""
	@echo "GPU:"
	@echo "  make gpu-up       Launch GPU instance with SkyPilot"
	@echo "  make gpu-down     Terminate GPU instance"

# Installation
install:
	uv pip install -e .

dev-install:
	uv pip install -e ".[dev]"
	pre-commit install

# Code quality
format:
	ruff format src/ tests/
	ruff check src/ tests/ --fix

lint:
	ruff check src/ tests/

type-check:
	mypy src/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache

# Docker services
docker-up:
	docker-compose -f docker/docker-compose.dev.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.dev.yml down

docker-logs:
	docker-compose -f docker/docker-compose.dev.yml logs -f

# GPU management (requires SkyPilot)
gpu-up:
	sky launch -c crown-gpu skypilot/gpu-a100.yaml

gpu-down:
	sky down crown-gpu

gpu-ssh:
	sky ssh crown-gpu

# Run examples
run-example:
	python src/scripts/train_example.py

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
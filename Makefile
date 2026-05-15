.PHONY: help install lint format test test-cov bench bench-regression bench-update security memory-profile clean proto docker-build docker-up docker-down run-local run-api run-distributed

help: ## Show this help
	@python -c "import re; lines = open('$(MAKEFILE_LIST)').readlines(); [print(f'\033[36m{m.group(1):20}\033[0m {m.group(2)}') for l in lines if (m := re.match(r'^([a-zA-Z_-]+):.*?## (.+)', l))]"

install: ## Install dependencies
	pip install -e ".[dev]"

install-cuda: ## Install with CUDA support
	pip install torch --index-url https://download.pytorch.org/whl/cu128
	pip install -e ".[dev]"

lint: ## Run linters
	ruff check src/distllm/ benchmarks/
	mypy src/distllm/

format: ## Format code
	black src/distllm/ benchmarks/
	ruff check --fix src/distllm/ benchmarks/

test: ## Run tests
	pytest -v

test-cov: ## Run tests with coverage
	pytest --cov=distllm --cov-report=term-missing --cov-report=html

bench: ## Run benchmarks
	python benchmarks/run.py --model roneneldan/TinyStories-1M

bench-large: ## Run benchmarks on larger model
	python benchmarks/run.py --model gpt2 --num-prompts 10 --max-tokens 100

bench-regression: ## Run pytest-benchmark tests and check for regressions
	python -m pytest tests/benchmark/ -v --benchmark-json=benchmarks/results.json -p no:asyncio
	python benchmarks/regression_check.py --baseline benchmarks/baseline.json --current benchmarks/results.json

bench-update: ## Update baseline with current benchmark results
	python -m pytest tests/benchmark/ -v --benchmark-json=benchmarks/results.json -p no:asyncio
	python -c "import json; results = json.load(open('benchmarks/results.json')); print('Update baseline manually with results from benchmarks/results.json')"

security: ## Run security scans (bandit, safety, detect-secrets)
	bash scripts/security_scan.sh

memory-profile: ## Run memory profiling tests
	python -m pytest tests/profiling/ -v --memory-profile -p no:asyncio

clean: ## Clean build artifacts
	python -c "import shutil, pathlib, os; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyo')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('*.egg-info')]; [shutil.rmtree(d, ignore_errors=True) for d in ['build', 'dist', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov'] if pathlib.Path(d).exists()]"

proto: ## Generate protobuf files
	python -m grpc_tools.protoc -I proto/ --python_out=src/distllm/communication/ --grpc_python_out=src/distllm/communication/ proto/node.proto

docker-build: ## Build Docker image
	docker build -t distributed-llm .

docker-up: ## Start with docker-compose
	docker-compose up

docker-down: ## Stop docker-compose
	docker-compose down

run-local: ## Run local mode
	distllm --model roneneldan/TinyStories-1M --local --chat

run-api: ## Run API server
	distllm-api --model roneneldan/TinyStories-1M --local --port 8000

run-distributed: ## Run distributed mode (starts all processes)
	scripts\start.bat

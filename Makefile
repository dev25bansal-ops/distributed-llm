.PHONY: help install lint format test test-cov bench bench-regression bench-update security memory-profile clean proto docker-build docker-up docker-down run-local run-api run-distributed helm-install helm-upgrade helm-template helm-lint kustomize-build-dev kustomize-build-staging kustomize-build-prod docker-build-multi docker-push sbom container-scan security-full pre-commit-install

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

# --- Helm ---
helm-install: ## Install Helm chart
	helm install distllm deploy/helm/ -f deploy/helm/values.yaml

helm-upgrade: ## Upgrade Helm release
	helm upgrade distllm deploy/helm/ -f deploy/helm/values.yaml --wait --timeout 10m

helm-template: ## Render Helm templates
	helm template distllm deploy/helm/ -f deploy/helm/values.yaml

helm-lint: ## Lint Helm chart
	helm lint deploy/helm/

# --- Kustomize ---
kustomize-build-dev: ## Build dev overlay manifests
	kustomize build deploy/kustomize/dev

kustomize-build-staging: ## Build staging overlay manifests
	kustomize build deploy/kustomize/staging

kustomize-build-prod: ## Build production overlay manifests
	kustomize build deploy/kustomize/production

# --- Docker multi-build ---
docker-build-multi: ## Build all CUDA variant images
	docker build -t distributed-llm:cuda12.8 -f Dockerfile --build-arg CUDA_VERSION=12.8.0 .
	docker build -t distributed-llm:cuda12.6 -f Dockerfile.cuda12.6 .
	docker build -t distributed-llm:cuda12.1 -f Dockerfile.cuda12.1 .

docker-push: ## Push images to registry
	docker tag distributed-llm:cuda12.8 $(REGISTRY)/distributed-llm:cuda12.8-$(TAG)
	docker tag distributed-llm:cuda12.6 $(REGISTRY)/distributed-llm:cuda12.6-$(TAG)
	docker tag distributed-llm:cuda12.1 $(REGISTRY)/distributed-llm:cuda12.1-$(TAG)
	docker push $(REGISTRY)/distributed-llm:cuda12.8-$(TAG)
	docker push $(REGISTRY)/distributed-llm:cuda12.6-$(TAG)
	docker push $(REGISTRY)/distributed-llm:cuda12.1-$(TAG)

# --- SBOM + Scanning ---
sbom: ## Generate CycloneDX SBOM
	cyclonedx-py -e . -o sbom.json --format json

container-scan: ## Scan Docker image with Trivy
	trivy image --severity HIGH,CRITICAL distributed-llm:cuda12.8

security-full: ## Run full security scan (bandit, safety, SBOM, container scan)
	$(MAKE) security
	$(MAKE) sbom
	$(MAKE) container-scan

# --- Pre-commit ---
pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type pre-push

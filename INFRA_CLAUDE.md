# CLAUDE.md — Infrastructure, CI/CD, Docs, Benchmarks

## Your Scope
You have ownership of root-level infrastructure: Docker, CI/CD, documentation, benchmarks, deployment manifests, scripts, and protobuf definitions.

## Do NOT Touch
- `src/distllm/` — any source code
- `tests/` — test files
- `integrations/` or `sdk/`

## Key Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Main Docker image (CUDA 12.8) |
| `Dockerfile.cuda12.1` | CUDA 12.1 variant |
| `Dockerfile.cuda12.6` | CUDA 12.6 variant |
| `docker-compose.yml` | Dev compose |
| `docker-compose.gpu.yml` | GPU compose |
| `docker-compose.pro.yml` | Production compose |
| `docker-entrypoint.sh` | Container entrypoint |
| `install.sh` | Installation script |
| `Makefile` | Build targets |
| `pyproject.toml` | Project config + coverage |
| `pytest.ini` | Pytest config |
| `requirements.txt` / `.lock` | Pinned dependencies |
| `.github/workflows/ci.yml` | CI pipeline (15 jobs) |
| `.github/workflows/release.yml` | Release pipeline |
| `.github/workflows/nightly-benchmark.yml` | Nightly benchmarks |
| `.github/workflows/nightly-loadtest.yml` | Nightly load tests |
| `.pre-commit-config.yaml` | Pre-commit hooks (pip-audit added) |
| `docs/BENCHMARKS.md` | Benchmark methodology (NEW) |
| `docs/ARCHITECTURE.md` | Architecture docs |
| `docs/CONFIG_REFERENCE.md` | Config reference |
| `deploy/` | Helm, Kustomize, Grafana, Prometheus |
| `benchmarks/` | Benchmark suite + Dockerfile |
| `proto/node.proto` | Protobuf definitions |
| `scripts/` | CI helper scripts, security audit |

## Current State
- Trivy pinned to `0.29.0`
- Coverage thresholds unified at 80%
- pip-audit in pre-commit
- Container scanning on every PR
- Dependabot for weekly updates
- Benchmark methodology published
- Containerized benchmark environment

## Commands
- `docker build -f Dockerfile -t distllm .` — build image
- `docker compose up -d` — start dev cluster
- `python -m benchmarks.run --model roneneldan/TinyStories-1M` — run benchmark
- `python -m benchmarks.cluster_benchmark --models llama8b --clusters 1 2` — cluster benchmark

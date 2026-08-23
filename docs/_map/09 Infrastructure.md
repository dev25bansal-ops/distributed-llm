---
tags:
  - infra
  - ci
  - docker
  - k8s
  - proto
aliases:
  - Infrastructure
  - Infrastructure & CI
---
# Infrastructure — CI/CD, deployment, packaging, gRPC contract & tooling

**Covers:** root infra (`pyproject.toml`, `Makefile`, Dockerfiles, docker-compose, devcontainer, install scripts), `deploy/` (k8s/helm/kustomize/operator/grafana/ray), `.github/workflows/` (23), `proto/`, and the operators' `benchmarks/` + `scripts/`.

> The operational surface: everything that builds, packages, ships, and reconciles DistLLM as running infrastructure rather than Python source.

## Packaging & build (root)

| file | LOC | purpose |
|------|-----|---------|
| `pyproject.toml` | 289 | setuptools src-layout, 30+ extras, console scripts (`distllm`, `distllm-coordinator`, `distllm-api`, `distllm-node`), ruff/black/mypy/pytest/coverage config |
| `Makefile` | 196 | install/lint/format/test/bench/cov/proto-gen/docker/helm/kustomize/load-test/chaos/security/tauri |
| `install.sh` | 200 | CUDA-aware one-command bootstrap (driver detect, nvidia-container-toolkit, docker build, compose, health wait) |
| `runner.sh` | 16 | manual dev scratch runner (not CI-referenced) |
| `Dockerfile` (+`cuda12.1`,`cuda12.6`) | 107/63/63 | multi-stage CUDA builds (variants are aliases) |
| `docker-compose.yml`/`.gpu.yml`/`.pro.yml` | 58/145/243 | baseline / scalable-worker GPU / production (LB + monitoring + auth profiles) |
| `.devcontainer/` | 66 | VS Code dev container (nvidia dd-in-ddc) |

## CI/CD — `.github/workflows/` (23 workflows + reusable)
`ci.yml` (main: ruff/mypy lint, test matrix, benchmark gates) · `release.yml` · `integration-test.yml` · `gpu-tests.yml` (self-hosted) · `load-testing.yml` · `memory-profile.yml` · `chaos-engineering.yml` · `nightly-{benchmark,loadtest,integration}.yml` · `quality-gates.yml` · `coverage/bandit/secrets` ratchets · `helm-publish.yml` · `deploy-website.yml` + `reusable-{ci,docker,deploy,release}.yml`.
- Pre-commit hooks (bandit + pip-audit), Dependabot.
- **Latest work:** E11 SLA tiers, `dist/` audit, scheduling wiring.

## `deploy/` — Kubernetes & observability manifests
| Area | Files | Purpose |
|------|-------|---------|
| `crds/` | `distllm.zeroroute.ai_crds.yaml` | CRD for `DistributedLLMCluster` |
| `helm/distllm-operator/` | Chart + CRDs + coordinator/worker/operator templates | run via Helm |
| `inference/` | coordinator deployment+PDB+network-policy+PVC, worker StatefulSet | raw manifests |
| `kustomize/{base,dev,staging,production}/` | base + environment patches + autoscaling | matrix overlay |
| `operator/controller.py` | 213 | **SCAFFOLD** polling k8s controller (reconcile is `logger.info` only) |
| `grafana/` `prometheus/` `loki/` | dashboards, alert rules, scrape/log configs | observability |
| `ray/` | autoscaler, serve_config, ci-workflow | Ray deployment path |
| `scripts/` | `backup.sh`, `setup-cron.sh` | ops |

## `proto/` — gRPC contract
- `node.proto` (167) — `NodeService`/`DraftModelService`: ForwardPass, HealthCheck, TransferWeights, `TensorProto`/`KVCacheProto`. Codegen via `make proto` → `src/distllm/communication` (`node_pb2*`).

## Operators' tooling
- **`benchmarks/`** (`run.py` 887, `cluster_benchmark.py`, `scaling_tests.py`, `competitive_benchmark.py`, `run_competitive.py`, `compare.py` vs vLLM/SGLang, `cost_routing_benchmark.py`, `ray_vs_grpc.py`, `regression_check.py`, `run_cpu_benchmarks.py` + `regression_config.json`/`baseline.json` + saved `results/*.json` + `HARDWARE_GUIDE.md`).
- **`scripts/`** — `install.sh`, Windows `.bat` launchers, `security_audit.py`/`security_scan.sh`, `bench_sla.py` (M13), `sla_tiers.py` (E11), plus `scripts/ci/` ratchet gates: `bandit_ratchet.py`, `coverage_ratchet.py`, `flaky_test_collector.py`/`flaky_test_ratchet.py`, `mutation_floor.py`, `run_correctness_cert.py`, `check_secrets_baseline.py` — each with a committed baseline JSON.

## Notes / dead code
- `deploy/operator/controller.py` is a **stub** (no k8s API calls — `_list_crs` returns empty unless `DISTLLM_CR_SAMPLE` set).
- Docker/kustomize/helm all target the same `DistributedLLMCluster` CR — parallel, non-shared paths.
- The K8s operator is effectively uncovered by tests.

## Tests
`tests/deploy/` (`test_gitops`, `test_helm`, `test_kustomize`, `test_operator`, `test_observability`), `tests/chaos/`, `tests/integration/`, `tests/benchmark/`, `tests/load/`, `tests/profiling/`; workflow-driven.
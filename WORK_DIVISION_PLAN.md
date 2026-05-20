# WORK DIVISION PLAN — 4 PARALLEL STREAMS

> **Zero file overlap between parts. No merge conflicts.**
> Each part owns a distinct set of files. Work on them in parallel with no issues.

---

## OVERVIEW — THE 4 PARTS

```
┌─────────────────────────────────────────────────────────────┐
│                    distributed-llm                           │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│   PART 1    │   PART 2     │   PART 3     │    PART 4       │
│  Core       │  API, CLI,   │  Infra &     │  Platform &     │
│  Engine     │  SDK         │  Deployment  │  Quality        │
│             │              │              │                 │
│ ~80 files   │ ~30 files    │ ~60 files    │ ~100 files      │
│ 7-10 days   │ 5-7 days     │ 5-7 days     │ 7-10 days       │
└─────────────┴──────────────┴──────────────┴─────────────────┘
```

---

## PART 1: DISTRIBUTED INFERENCE ENGINE

**Goal**: Fix the core engine so distributed inference actually works correctly and reliably.
**Estimated time**: 7-10 days
**Priority**: **HIGHEST** — everything depends on this

### Owned Files

| Directory | Files |
|-----------|-------|
| `src/distllm/core/` | All core engine files (134 entries) |
| `src/distllm/communication/` | gRPC layer (10 files) |
| `src/distllm/backends/` | Inference backends (4 files) |
| `src/distllm/models/` | Model management (4 files) |
| `src/distllm/config/` | Configuration (3 files) |
| `proto/` | Protobuf definitions (1 file) |
| `src/distllm/constants.py` | Named constants (1 file) |

### Tests You Own

| Directory | Files |
|-----------|-------|
| `tests/core/` | 45 test files |
| `tests/communication/` | 2 test files |
| `tests/integration/` | 5 test files (pipeline, distributed, streaming, node lifecycle, KV gossip) |
| `tests/property/` | test_kv_cache, test_batch_scheduler, test_model_configs, test_speculative_decoder |
| `tests/correctness/` | 5 files (numerical stability, KV cache, output quality, quantization, speculative) |
| `tests/compression/` | 1 file |

---

### 📋 WORK ITEMS — PART 1



#### P1-D: Communication Layer Fixes (Priority: 🟡 MEDIUM)

| # | Task | File | Details |
|---|------|------|---------|
| 16 | Validate protobuf message size limits | `communication/node_service.py` | gRPC has 4MB default message limit. Add configurable max message size for large tensor transfers |

#### P1-H: Test Fixes (Priority: 🟡 MEDIUM)

| # | Task | File | Details |
|---|------|------|---------|
| 28 | Add real model inference test | Create `tests/core/test_real_inference.py` | Download a small model (e.g., Phi-2) and run a real forward pass through the pipeline. This is the ONLY way to verify inference actually works |



#### ✅ PART 1 — DONE CHECKLIST

- [x] Task 1: Speculative decoder doesn't crash
- [x] Task 2: Top-p sampling produces correct outputs
- [x] Task 3: Batch scheduler doesn't drop requests
- [x] Task 4: KV cache is thread-safe
- [x] Task 5: All shared state has proper locking
- [x] Task 6-8: Coordinator refactored to clean architecture
- [x] Task 9-12: Pipeline, streaming, recovery, prefill all work
- [x] Task 13-15,17: gRPC has timeouts, retries, proper serialization, NCCL fallback
- [x] Task 18-20: All 3 backends (PyTorch, vLLM, llama.cpp) work
- [x] Task 21-23: Model loading/partition works for all architectures
- [x] Task 24-26: Configuration validates correctly
- [x] Task 27,29,30: All owned tests fixed
- [ ] Task 28: Real model inference test added
- [x] Task 31-34: CUDA graphs, FP8, unused imports cleanup, and comprehensive logging done
- [ ] Task 16: Validate protobuf message size limits

**Final verification**: `pytest tests/core/ tests/communication/ tests/integration/ tests/correctness/ -v --timeout=60`

---

## PART 2: API, CLI & SDK

**Goal**: Make the user-facing interfaces production-ready — real endpoints, proper errors, working CLI.
**Estimated time**: 5-7 days
**Priority**: **HIGH** — this is what users actually interact with

### Owned Files

| Directory | Files |
|-----------|-------|
| `src/distllm/api/` | FastAPI server, routes, middleware (~20 files) |
| `src/distllm/api/routes/` | All route handlers (22 files) |
| `src/distllm/cli/` | Typer CLI (12 files) |
| `src/distllm/sdk/` | Python client SDK (4 files) |
| `src/distllm/errors/` | Error types and policies (3 files) |
| `src/distllm/utils/` | Utility functions (2 files) |
| `src/distllm/__init__.py` | Package exports (1 file) |

### Tests You Own

| Directory | Files |
|-----------|-------|
| `tests/api/` | 9 test files |
| `tests/sdk/` | 1 test file |
| `tests/prompts/` | 1 test file |
| Root: `test_plugin_marketplace.py` | (shared, assigned here) |

---

### 📋 WORK ITEMS — PART 2

#### P2-B: Clean Up Stub Routes (Priority: 🔴 HIGH)

| # | Task | File | Details |
|---|------|------|---------|
| 7 | Remove or implement image generation endpoint | `api/routes/images.py` | Returns 501. Either implement a minimal version (even if it just calls a simple model) OR remove the route and add `include_in_schema=False` |
| 8 | Remove or implement audio endpoint | `api/routes/audio.py` | Returns 501. Same approach as images |

#### ✅ PART 2 — DONE CHECKLIST

- [x] Task 1-6: Server startup, CORS, lifecycle, middleware all fixed
- [x] Task 9-17: All 501 stub routes handled (moderations, files, batch, RAG, agent, disagg, gossip, optimization, debug) — implemented or hidden
- [x] Task 18: Fine-tuning API wired to real training code
- [x] Task 19-24: Chat, completion, embeddings, health, model list all correct
- [x] Task 25-28: Auth, security headers, request limits working
- [x] Task 29-35: CLI commands all work with proper output
- [x] Task 36-40: SDK has timeouts, retries, proper errors, circuit breaker
- [x] Task 41-43: Error responses match OpenAI format
- [x] Task 44-45: Package exports clean, utilities work
- [x] Task 46-49: All owned tests pass
- [ ] Task 7: Image generation endpoint (still returns 501 in `api/routes/images.py`)
- [ ] Task 8: Audio endpoint (still returns 501 in `api/routes/audio.py`)

**Final verification**: `pytest tests/api/ tests/sdk/ tests/prompts/ -v --timeout=30`

---

## PART 3: INFRASTRUCTURE & DEPLOYMENT

**Goal**: Make deployment real (not theatre). Docker images publishable, Helm charts that actually deploy, real infrastructure that works.
**Estimated time**: 5-7 days
**Priority**: **MEDIUM** — needed for production but not for demo

### Owned Files

| Directory | Files |
|-----------|-------|
| `deploy/helm/` | Helm chart (25+ files) |
| `deploy/kustomize/` | Kustomize overlays (15+ files) |
| `deploy/inference/` | Raw K8s manifests (5 files) |
| `deploy/grafana/` | Grafana dashboards (2 files) |
| `deploy/gitops/` | ArgoCD, Flux configs (5 files) |
| `deploy/karpenter/` | Karpenter configs (4 files) |
| `deploy/operator/` | Operator manifests (3 files) |
| `deploy/webhook/` | Webhook manifests (3 files) |
| `src/distllm/deploy/` | Canary, rollout, version mgmt (7 files) |
| `src/distllm/operator/` | K8s operator (7 files) |
| `src/distllm/cloud/` | Cloud provider integration (9 files) |
| `src/distllm/cloud/providers/` | AWS, Azure, GCP (3 files) |
| `src/distllm/router/` | Multi-cluster routing (5 files) |
| `src/distllm/gateway/` | API gateway (4 files) |
| `src/distllm/scheduling/` | Cost-aware scheduling (4 files) |
| `src/distllm/plugins/` | Plugin system (6 files) |
| Root: `Dockerfile`, `Dockerfile.cuda12.1`, `Dockerfile.cuda12.6` |
| Root: `docker-compose.yml` |
| Root: `docker-entrypoint.sh` |
| Root: `install.sh` |
| `scripts/` | Utility scripts (5 files) |

### Tests You Own

| Directory | Files |
|-----------|-------|
| `tests/gateway/` | 2 test files |
| `tests/features/` | 4 test files |

---

### 📋 WORK ITEMS — PART 3

#### P3-D: Infrastructure Theatre Removal (Priority: 🔴 HIGH)

| # | Task | File | Details |
|---|------|------|---------|
| 31 | Remove or mark Karpenter configs | `deploy/karpenter/` | These are advanced K8s features not needed for MVP. Remove all files and add to roadmap |
| 32 | Remove or mark ArgoCD configs | `deploy/gitops/argocd/` | Not needed for MVP. Remove |
| 33 | Remove or mark Flux configs | `deploy/gitops/flux/` | Not needed for MVP. Remove |

#### ✅ PART 3 — DONE CHECKLIST

- [x] Task 1-10: Docker builds, multi-stage, non-root, HEALTHCHECK, CUDA variants, docker-compose, entrypoint, .dockerignore, GHCR publish all done
- [x] Task 11-26: Helm chart templates fixed, values.yaml updated, `_helpers.tpl` improved
- [x] Task 27-30: All Kustomize overlays (base, dev, staging, production) fixed
- [x] Task 34-36: Grafana dashboards kept with real content, webhook/inference manifests real
- [x] Task 37-41: Operator verified (controllers, GPU scheduler, CRDs)
- [x] Task 42-48: Cloud providers real (AWS, Azure, GCP), spot handling, auto-provision, budget, migration
- [x] Task 49-54: Router and gateway real (multi-cluster, discovery, consistent hash, backend, fallback)
- [x] Task 55-58: Plugin system real (installer, sandbox, compatibility, telemetry)
- [x] Task 59-64: Deploy and scheduling code real (canary, rollout, version mgmt, cost, spot price, install.sh)
- [x] Task 65-66: All owned tests (gateway, features) pass
- [ ] Task 31: Remove Karpenter configs (`deploy/karpenter/` still exists)
- [ ] Task 32: Remove ArgoCD configs (`deploy/gitops/argocd/` still exists)
- [ ] Task 33: Remove Flux configs (`deploy/gitops/flux/` still exists)

**Final verification**: `helm lint deploy/helm/ && docker build -t distllm:test . && docker-compose config && pytest tests/gateway/ tests/features/ -v --timeout=30`

---

## PART 4: PLATFORM & QUALITY

**Goal**: Observability works, dashboard is real, edge is real or removed, project files are professional, CI/CD actually gates.
**Estimated time**: 7-10 days
**Priority**: **MEDIUM** — important for polish and YC readiness

### Owned Files

| Directory | Files |
|-----------|-------|
| `src/distllm/observability/` | Logging, metrics, tracing (6 files) |
| `src/distllm/monitoring/` | Alerting, anomaly detection (4 files) |
| `src/distllm/health/` | Health checks, failover (5 files) |
| `src/distllm/dashboard/` | Web monitoring dashboard (2 files) |
| `src/distllm/ui/` | Web UI — Jinja2 templates (2 files) |
| `src/distllm/tenant/` | Multi-tenancy (6 files) |
| `src/distllm/edge/` | Edge deployment (5 files) |
| `src/distllm/prompts/` | Prompt templates (2 files) |
| `src/distllm/profiling/` | Profiling tools (2 files) |
| `src/distllm/benchmarks/` | Integrated benchmarks (2 files) |
| `src/distllm/chaos/` | Chaos testing (2 files) |
| Root: `.github/workflows/` | CI/CD pipelines (8 files) |
| Root: `.pre-commit-config.yaml` | Pre-commit hooks |
| Root: `Makefile` | Build automation |
| Root: `pyproject.toml` | Package config |
| Root: `pytest.ini` | Test config |
| Root: `requirements.txt`, `requirements.lock` | Dependencies |
| Root: `.gitignore`, `.dockerignore` | Exclusion files |
| Root: `.secrets.baseline` | Secrets detection |
| Root: `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `DEPLOYMENT.md`, `LICENSE` | Documentation |
| Root: `config.yaml` | Default config |
| `integrations/` | LangChain, LlamaIndex packages |
| `examples/` | Usage examples |
| `extensions/vscode/` | VS Code extension |
| `benchmarks/` | Benchmark scripts |

### Tests You Own

| Directory | Files |
|-----------|-------|
| `tests/e2e/` | 10+ test files |
| `tests/chaos/` | 7+ test files |
| `tests/load/` | 9+ test files + 5 scenarios |
| `tests/fuzz/` | 5 test harnesses |
| `tests/security/` | 2 test files |
| `tests/benchmark/` | 1 test file |
| `tests/dashboard/` | 1 test file |
| `tests/edge/` | 1 test file |
| `tests/mutation/` | 1 test file |
| `tests/profiling/` | 2 test files |
| `tests/models/` | 2 test files |
| Root tests: `test_pipeline.py`, `test_distributed.py`, `test_cloud_cost.py`, `test_federation.py`, `test_speculative_auto.py` |
| `tests/conftest.py` | Shared fixtures (coordinate changes with other parts) |

---

### 📋 WORK ITEMS — PART 4



#### ✅ PART 4 — DONE CHECKLIST

- [x] Task 1-7: Observability working correctly (tracing, metrics, logging, Loki)
- [x] Task 9,11-12,14: WebSocket cleanup, graceful startup, UI port 8500, auto-refresh working
- [x] Task 15-20: Alerting, monitoring, anomaly detection, health probes, failover all working
- [x] Task 21-28: Edge module has REAL inference (quantized, routing, sharding, serving)
- [x] Task 29-33: Multi-tenancy verified (store, router, middleware, rate limiter, billing)
- [x] Task 35-40,43: Documentation accurate (README, CHANGELOG, CONTRIBUTING, DEPLOYMENT, config.yaml)
- [x] Task 44-52: CI/CD pipelines gate on failures (tests, lint, security, secrets, benchmark, container scan, release, Python matrix)
- [x] Task 54-57: Pre-commit hooks work (ruff, mypy, trailing-whitespace), Makefile cleaned
- [x] Task 60-64: Package config correct (pyproject.toml, requirements.lock 3.10-3.12, test dep group, markers)
- [x] Task 65-72: Integrations verified (LangChain, LlamaIndex, SDK examples, VS Code extension)
- [x] Task 73-78: Benchmarks and profiling work (runner, cluster, compare, regression, CI profiler)
- [x] Task 79-92,93-95: e2e, fuzz, load, chaos, security, mutation, property tests, pytest-timeout all done
- [x] Task 8,10,13: Dashboard fixes done (ws_handler cleaned, port 8501, UI client reused)
- [x] Task 34: Tenant API tests added (`tests/api/test_tenants.py`)
- [x] Task 41-42: CODE_OF_CONDUCT.md and SECURITY.md added
- [x] Task 53: OS matrix includes Windows (`windows-latest`)
- [x] Task 58-59: `make test-all` and `make pre-commit-run` targets exist

**Final verification**: `pytest tests/e2e/ tests/chaos/ tests/fuzz/ tests/security/ tests/dashboard/ tests/edge/ -v --timeout=120`

---

## FILE OWNERSHIP SUMMARY (Zero Overlap)

```
src/distllm/
├── __init__.py              → PART 2
├── constants.py             → PART 1
├── api/                     → PART 2
├── backends/                → PART 1
├── benchmarks/              → PART 4
├── chaos/                   → PART 4
├── cli/                     → PART 2
├── cloud/                   → PART 3
├── communication/           → PART 1
├── config/                  → PART 1
├── core/                    → PART 1
├── dashboard/               → PART 4
├── deploy/                  → PART 3
├── edge/                    → PART 4
├── errors/                  → PART 2
├── gateway/                 → PART 3
├── health/                  → PART 4
├── models/                  → PART 1
├── monitoring/              → PART 4
├── observability/           → PART 4
├── operator/                → PART 3
├── plugins/                 → PART 3
├── profiling/               → PART 4
├── prompts/                 → PART 4
├── router/                  → PART 3
├── scheduling/              → PART 3
├── sdk/                     → PART 2
├── tenant/                  → PART 4
├── ui/                      → PART 4
├── utils/                   → PART 2

tests/
├── conftest.py              → PART 4 (shared — coordinate changes)
├── api/                     → PART 2
├── benchmark/               → PART 4
├── chaos/                   → PART 4
├── communication/           → PART 1
├── compression/             → PART 1
├── core/                    → PART 1
├── correctness/             → PART 1
├── dashboard/               → PART 4
├── e2e/                     → PART 4
├── edge/                    → PART 4
├── features/                → PART 3
├── fuzz/                    → PART 4
├── gateway/                 → PART 3
├── integration/             → PART 1
├── load/                    → PART 4
├── models/                  → PART 1
├── mutation/                → PART 4
├── profiling/               → PART 4
├── prompts/                 → PART 2
├── property/                → PART 1 (core tests) + PART 4 (remaining)
├── sdk/                     → PART 2
├── security/                → PART 4

Root files
├── Dockerfile*              → PART 3
├── docker-compose.yml       → PART 3
├── docker-entrypoint.sh     → PART 3
├── install.sh               → PART 3
├── Makefile                 → PART 4
├── pyproject.toml           → PART 4
├── pytest.ini               → PART 4
├── requirements.txt         → PART 4
├── requirements.lock        → PART 4
├── config.yaml              → PART 4
├── .github/                 → PART 4
├── .pre-commit-config.yaml  → PART 4
├── .gitignore               → PART 4
├── .dockerignore            → PART 4
├── .secrets.baseline        → PART 4
├── README.md                → PART 4
├── CONTRIBUTING.md          → PART 4
├── CHANGELOG.md             → PART 4
├── DEPLOYMENT.md            → PART 4
├── LICENSE                  → PART 4
├── proto/                   → PART 1
├── deploy/                  → PART 3
├── integrations/            → PART 4
├── examples/                → PART 4
├── extensions/              → PART 4
├── scripts/                 → PART 3
├── benchmarks/              → PART 4
```

---

## WORKFLOW RULES

### To Avoid Merge Conflicts

1. **Never edit a file owned by another part** — if you need to change it, ask the owner
2. **Coordinate conftest.py changes** — it's the only shared file. Use brief communication before editing
3. **Own `__init__.py` means you control exports** — Part 2 owns the package's public API surface
4. **Part 1 exports are the contract** — if Part 2 needs a new export from core, Part 1 must add it

### Recommended Order

```
Week 1-2:  Part 1 (Core Engine — fixes critical bugs first)
           Part 2 (API/CLI/SDK — can start once core is stable)

Week 2-3:  Part 1 continues
           Part 2 continues
           Part 3 (Infrastructure — needs working core + API)

Week 3-4:  Part 1 wrapping up
           Part 2 wrapping up
           Part 3 continues
           Part 4 (Platform & Quality — needs everything else stable)

Week 4+:   Integration testing, bug fixes, YC application prep
```

### How to Mark Work Complete

Each task is checkbox-able. At the end of each part's checklist, run the final verification command. If all tests pass and all checkboxes are checked, the part is done.

### Parallel Work is Safe

| Working at same time | Conflict risk |
|---------------------|---------------|
| Part 1 + Part 2 | ✅ **None** — different directories |
| Part 1 + Part 3 | ✅ **None** — source vs deployment |
| Part 2 + Part 4 | ✅ **None** — API vs observability |
| Part 3 + Part 4 | ✅ **None** — infra vs quality |
| All 4 parts | ✅ **Safe** — zero file overlap |

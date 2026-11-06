---
tags:
  - meta
  - index
  - completeness
---
# Module Index — every folder in the repository

> This is the **completeness gate**: every top-level source package and every major repository folder is listed below with a one-line purpose and a link to the note that covers it. If a folder is not here, it is generated/build/cache output and is intentionally excluded from the vault graph (see [[_Project Overview#Vault layout & conventions]]).

## 1 · `src/distllm/` — the engine (all packages)

All entries `src/distllm/<pkg>`. Full per-file maps live in the linked MOC.

| Package | # | LOC | Purpose | Covering note |
|---------|---|-----|---------|---------------|
| `core/` | 274 | 75,171 | Coordinator, batch scheduler, KV cache, speculative decoding, routers, cost/billing, HA, multimodal, plugins | [[01 Core Engine]] |
| `dist/` | 194 | 71,033 | Distributed execution plane: pipeline, p2p/gossip, partition, topology, federation, chaos, NAT/ICE, disagg | [[02 Distributed Layer]] |
| `api/` | 88 | 23,851 | FastAPI OpenAI-compatible server: routes, middleware, SSO/RBAC, WAF, rate-limit | [[03 API Server]] |
| `cli/` | 50 | 10,182 | Typer CLI (`distllm`) command groups | [[04 CLI & Config]] |
| `config/` | 18 | 2,218 | pydantic-settings `DistLLMSettings` + loader/resolver/profiles/heuristics | [[04 CLI & Config]] |
| `client/` | 2 | 450 | Async/sync REST client (public SDK entry) | [[04 CLI & Config]] |
| `backends/` | 19 | 5,855 | `BackendAdapter` contract + vLLM/llama.cpp/ONNX/Triton/NIM/… adapters | [[05 Backends & Models]] |
| `models/` | 10 | 2,786 | Model hub, partitioner, adapter (LoRA), RoPE scaling | [[05 Backends & Models]] |
| `security/` | 16 | 4,030 | E2E encrypt, watermark, content moderation, attestation scaffolds | [[06 Security]] |
| `compliance/` | 2 | 626 | Auditor evidence packs (SOC2/ISO/GDPR/HIPAA/Export) | [[06 Security]] |
| `integrations/` | 12 | 2,193 | in-tree MLflow/Spark/WandB + CI integrations + `_common` | [[07 Integrations]] |
| `prompts/` | 17 | 2,984 | Prompt library, template engine, exchange marketplace | [[07 Integrations]] |
| `plugins/` | 9 | 1,913 | Hook plugin system + built-ins + CI (Airflow/KFP/MLflow) | [[07 Integrations]] |
| `sdk/` | 7 | 1,879 | Public Python client SDK (async/sync) + streaming/transport | [[07 Integrations]] |
| `observability/` | 20 | 11,755 | Prometheus/OTel/Loguru, SLO, capacity planning, incident, self-healing | [[11 Platform Services]] |
| `health/` | 5 | 301 | Node health probes/state/failover | [[11 Platform Services]] |
| `cache/` | 3 | 1,009 | CRDT cross-cluster prefix cache, digest/gossip | [[11 Platform Services]] |
| `cloud/` | 7 | 2,455 | AWS/GCP/Azure pricing + spot orchestrator + control-plane agent | [[11 Platform Services]] |
| `errors/` | 4 | 963 | Error hierarchy + retry policies | [[11 Platform Services]] |
| `utils/` | 4 | 177 | GBNF grammar, lazy-import, scheduling helpers | [[11 Platform Services]] |
| `terraform/` | 1 | 225 | Terraform IaC resources → coordinator REST | [[11 Platform Services]] |
| `verification/` | 6 | 1,541 | Correctness comparator, hash registry, report | [[11 Platform Services]] |
| `dashboard/` | 3 | 681 | WebSocket metrics dashboard (v1/v2) | [[11 Platform Services]] |
| `ui/` | 2 | 74 | FastAPI Jinja web UI over coordinator | [[11 Platform Services]] |
| `benchmarks/` | 3 | 302 | scaling/throughput harness + lm-eval integration | [[11 Platform Services]] |
| `constants.py` `errors/`… | — | — | root singletons | [[00 Module Index]] §this file |

<!-- recounted 2026-11-06 by hygiene pass -->

## 2 · Support trees at repo root

| Folder | Contents | Map |
|--------|----------|-----|
| `tests/` | 788 test files, ~152K LOC, ~51 packages | [[10 Tests]] |
| `integrations/` | 168 files / 24 framework adapters + IaC: agno→spring-ai, ansible→kubectl| [[07 Integrations]] |
| `sdk/` | 61 | multi-language client SDKs: python/go/js/rust + openapi | [[07 Integrations]] |
| `apps/` (removed 2026-11-06 hygiene pass) | chat / RAG / multi-agent demo apps | [[08 Frontends]] |
| `examples/` | 20 | integration scripts + notebooks | [[08 Frontends]] |
| `extensions/vscode/` | 17 | VS Code extension | [[08 Frontends]] |
| `website/` | 137 | static marketing/product site | [[08 Frontends]] |
| `tauri/` | 92 | desktop app (Svelte+tauri+Rust) | [[08 Frontends]] |
| `benchmarks/` | 8 | perf/scaling/competitive harness | [[09 Infrastructure]] |
| `scripts/` | 23 | installer, Windows launchers, security audit, CI ratchets | [[09 Infrastructure]] |
| `deploy/` | 69 | k8s helm/kustomize/crds/operator, grafana, ray | [[09 Infrastructure]] |
| `proto/` | 1 | gRPC `node.proto` wire contract | [[09 Infrastructure]] |
| `.github/` | 33 | 25 CI workflows + reusable | [[09 Infrastructure]] |
| `docs/` | 99 files (98 .md) | **this vault** + reference docs | [[_Project Overview]] [[09 Infrastructure]] |

<!-- recounted 2026-11-06 by hygiene pass -->

## 3 · Deliberately excluded folders (not project source)

`node_modules/`, `build/`, `dist/`, `.venv*/`, `.git/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.hypothesis/`, `certs/`, `.distllm*/`, `.agent/`, `.claude/` internals, `__pycache__/`.

> **Stray vault stubs (removed):** `distllm/` and `distributed-llm/` at repo root were accidental empty Obsidian starter vaults (only `Welcome.md` + default config); they were deleted in the 2026-11-06 hygiene pass and no longer exist.
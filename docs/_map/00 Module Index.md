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
| `core/` | 296 | 81,744 | Coordinator, batch scheduler, KV cache, speculative decoding, routers, cost/billing, HA, multimodal, plugins | [[01 Core Engine]] |
| `dist/` | 193 | 67,348 | Distributed execution plane: pipeline, p2p/gossip, partition, topology, federation, chaos, NAT/ICE, disagg | [[02 Distributed Layer]] |
| `api/` | 89 | 21,832 | FastAPI OpenAI-compatible server: routes, middleware, SSO/RBAC, WAF, rate-limit | [[03 API Server]] |
| `cli/` | 44 | 8,478 | Typer CLI (`distllm`) command groups | [[04 CLI & Config]] |
| `config/` | 18 | 2,113 | pydantic-settings `DistLLMSettings` + loader/resolver/profiles/heuristics | [[04 CLI & Config]] |
| `client/` | 2 | 450 | Async/sync REST client (public SDK entry) | [[04 CLI & Config]] |
| `backends/` | 18 | 4,809 | `BackendAdapter` contract + vLLM/llama.cpp/ONNX/Triton/NIM/… adapters | [[05 Backends & Models]] |
| `models/` | 10 | 2,735 | Model hub, partitioner, adapter (LoRA), RoPE scaling | [[05 Backends & Models]] |
| `security/` | 16 | 3,710 | E2E encrypt, watermark, content moderation, attestation scaffolds | [[06 Security]] |
| `compliance/` | 2 | 626 | Auditor evidence packs (SOC2/ISO/GDPR/HIPAA/Export) | [[06 Security]] |
| `integrations/` | 12 | 2,193 | in-tree MLflow/Spark/WandB + CI integrations + `_common` | [[07 Integrations]] |
| `prompts/` | 17 | 2,984 | Prompt library, template engine, exchange marketplace | [[07 Integrations]] |
| `plugins/` | 9 | 1,815 | Hook plugin system + built-ins + CI (Airflow/KFP/MLflow) | [[07 Integrations]] |
| `sdk/` | 7 | 1,773 | Public Python client SDK (async/sync) + streaming/transport | [[07 Integrations]] |
| `observability/` | 20 | 11,749 | Prometheus/OTel/Loguru, SLO, capacity planning, incident, self-healing | [[11 Platform Services]] |
| `health/` | 5 | 301 | Node health probes/state/failover | [[11 Platform Services]] |
| `cache/` | 3 | 1,009 | CRDT cross-cluster prefix cache, digest/gossip | [[11 Platform Services]] |
| `cloud/` | 7 | 2,427 | AWS/GCP/Azure pricing + spot orchestrator + control-plane agent | [[11 Platform Services]] |
| `errors/` | 4 | 899 | Error hierarchy + retry policies | [[11 Platform Services]] |
| `utils/` | 4 | 177 | GBNF grammar, lazy-import, scheduling helpers | [[11 Platform Services]] |
| `terraform/` | 1 | 225 | Terraform IaC resources → coordinator REST | [[11 Platform Services]] |
| `verification/` | 6 | 1,541 | Correctness comparator, hash registry, report | [[11 Platform Services]] |
| `dashboard/` | 3 | 681 | WebSocket metrics dashboard (v1/v2) | [[11 Platform Services]] |
| `ui/` | 2 | 74 | FastAPI Jinja web UI over coordinator | [[11 Platform Services]] |
| `benchmarks/` | 3 | 303 | scaling/throughput harness + lm-eval integration | [[11 Platform Services]] |
| `constants.py` `errors/`… | — | — | root singletons | [[00 Module Index]] §this file |

## 2 · Support trees at repo root

| Folder | Contents | Map |
|--------|----------|-----|
| `tests/` | 812 test files, ~229K LOC, ~48 packages | [[10 Tests]] |
| `integrations/` | 28 framework adapters + IaC: agno→spring-ai, ansible→kubectl| [[07 Integrations]] |
| `sdk/` | multi-language client SDKs: python/go/js/rust + openapi | [[07 Integrations]] |
| `apps/` | chat / RAG / multi-agent demo apps | [[08 Frontends]] |
| `examples/` | integration scripts + notebooks | [[08 Frontends]] |
| `extensions/vscode/` | VS Code extension | [[08 Frontends]] |
| `website/` | static marketing/product site | [[08 Frontends]] |
| `tauri/` | desktop app (Svelte+tauri+Rust) | [[08 Frontends]] |
| `benchmarks/` | perf/scaling/competitive harness | [[09 Infrastructure]] |
| `scripts/` | installer, Windows launchers, security audit, CI ratchets | [[09 Infrastructure]] |
| `deploy/` | k8s helm/kustomize/crds/operator, grafana, ray | [[09 Infrastructure]] |
| `proto/` | gRPC `node.proto` wire contract | [[09 Infrastructure]] |
| `.github/` | 23 CI workflows + reusable | [[09 Infrastructure]] |
| `docs/` | **this vault** + reference docs (70 .md) | [[_Project Overview]] [[09 Infrastructure]] |

## 3 · Deliberately excluded folders (not project source)

`node_modules/`, `build/`, `dist/`, `.venv*/`, `.git/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.hypothesis/`, `certs/`, `.distllm*/`, `.agent/`, `.claude/` internals, `__pycache__/`.

> **Stray vault stubs:** `distllm/` and `distributed-llm/` at repo root are accidental empty Obsidian starter vaults (only `Welcome.md` + default config). They are not part of the real vault and are candidates for deletion.
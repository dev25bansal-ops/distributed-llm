---
tags:
  - cli
  - config
  - client
aliases:
  - CLI & Config
---
# CLI, Config & Client — `src/distllm/{cli,config,client}/`

**64 .py files · ~13.1K LOC** (`cli/` 44 · `config/` 18 · `client/` 2).

> The operator surface: the all-in-one **Typer CLI** (`distllm`), the full **pydantic-settings configuration model** (`DistLLMSettings`, ~60 nested settings classes), and the public **async/sync Python client** for the coordinator REST API.
>
> **Tests:** `tests/cli/`, `tests/config/`, `tests/client/` + `tests/sdk/test_client.py` (`python -m pytest tests/cli tests/config tests/client -v`).

## `cli/` — the `distllm` CLI (44 files)

| file | LOC | purpose |
|------|-----|---------|
| `main.py` | 1,527 | the `distllm` typer `app`; registers all sub-groups (`model/cluster/config/system/router/security/defrag/federate/daas/draft/tune`) |
| `__init__.py` | 14 | re-exports `app`, `main` |
| `cluster.py` | 357 | `_cluster_start` (launch coord+worker), `join/leave/list_nodes/status/scale/drain/rebalance` |
| `doctor.py` | 585 | `Doctor` — CUDA/GPU/net/perf/port/model-compat/config/disk/python-env diagnostics |
| `onboard.py` | 569 | `run_onboard` — first-run setup wizard (hardware detect, model recommend, config gen) |
| `chat_stream.py` | 420 | interactive streaming chat REPL (slash handlers, key bindings) |
| `tune.py` | 416 | quantization/batch/cache tuning |
| `observe.py` | 274 | Grafana dashboard/datasource generators + metrics server |
| `autopsy.py` | 246 | full cluster crash/inspection dump |
| `output.py` | 363 | rich-formatting hub (table/tree/panel/json) |
| `client.py` | 188 | CLI-side `DistLLMClient`/`ClientConfig` (internal HTTP client) |
| `profile.py` | 132 | real-inference perf profiling |
| `benchmark.py` | 201 | run/compare benchmarks |
| `deploy.py` | 222 | `run_deploy` + estimation |
| `models.py` | 61 | `CoordinatorArgs`/`ApiServerArgs`/`WorkerArgs` launch flags |
| `adapters.py` | 74 | LoRA adapter manage over HTTP |
| `backup.py` | 97 | config/cluster backups |
| `cert.py` | 71 | TLS cert create/info/renew/revoke |
| `chat.py` | 63 | basic non-streaming chat |
| `completion.py` | 67 | shell-completion script generation |
| `compress.py` | 81 | offline weight compression |
| `config_commands.py` | 54 | `config_validate/reference/openapi` |
| `cost_avoid.py` | 142 | calculate cost avoidance |
| `daas_commands.py` | 103 | `daas_serve/status/benchmark` |
| `defrag_commands.py` | 109 | `defrag_status/run/stats` |
| `draft_commands.py` | 69 | draft-fleet status |
| `error_handler.py` | 148 | CLI error handling decorator |
| `eval.py` | 137 | eval runner |
| `federate_commands.py` | 104 | federated train/save/deploy/status |
| `install.py` | 160 | preflight, download test model, verify install |
| `logs.py` | 73 | stream/log entries |
| `notify.py` | 76 | send/history notifications |
| `prompts.py` | 118 | prompt list/show/categories/use |
| `profile_presets.py` | 104 | profile presets |
| `prompts.py` | — | (see above) |
| `quota.py` | 189 | quota invoice/report/export |
| `router.py` | 107 | router rules/dry-run/stats |
| `run.py` | 49 | one-shot inference |
| `setup_wizard.py` | 153 | interactive config authoring |
| `setup.py` | 86 | one-shot config writer |
| `status.py` | 51 | live cluster status |
| `system_commands.py` | 213 | launchers (coordinator/api/observe) |
| `tutorial.py` | 106 | guided tutorial |
| `verify.py` | 31 | backend compat checks |
| `webhook.py` | 68 | webhook register/list/test |

## `config/` — the configuration model (18 files)

| file | LOC | purpose |
|------|-----|---------|
| `settings.py` | 416 | `DistLLMSettings` (pydantic BaseSettings, `DISTLLM_` prefix + `__` nesting); aggregates all domain settings |
| `resolver.py` | 136 | `ConfigResolver` — locate a config file |
| `loader.py` | 136 | config-loading + `*Config` aliases → `*Settings` |
| `_model.py` | 227 | model/quantization/speculative/LoRA/MoE/multi-model/compression/model-hub/embedding settings |
| `_network.py` | 159 | coordinator/network/TLS/rate-limit/wide-area/router-rule settings |
| `_parallelism.py` | 138 | node/tensor-parallel/partitioning/rebalancer/batching/chunked-prefill/disagg |
| `model_heuristics.py` | 133 | name-based parameter/VRAM heuristics |
| `profiles.py` | 142 | `ProfileConfig` — latency/throughput/memory presets |
| `reference.py` | 87 | markdown config-reference generator |
| `_cache.py` | 99 | prefix-cache/cache-persistence/gossip/cache settings |
| `_deployment.py` | 79 | rollout/canary/version/cost/tenant settings |
| `_backends.py` | 67 | VLLM/Llama.cpp settings |
| `_hardware.py` | 34 | device/GPU topology |
| `_observability.py` | 32 | monitoring/alerting/chaos |
| `_performance.py` | 36 | CUDA-graph/compile/adaptive-precision/self-optimizing |
| `_application.py` | 32 | RAG/agent/plugin settings |
| `_generation.py` | 29 | sampling defaults |
| `__init__.py` | 78 | `DistLLMSettings` facade + loader/resolver |

## `client/` — the CLI/internal REST client (2 files)

| file | LOC | purpose |
|------|-----|---------|
| `client.py` | 423 | `DistLLMClient` (async HTTP SDK): connect/generate/chat/stream/list_models/nodes/metrics; `SyncDistLLMClient`; response dataclasses |
| `__init__.py` | 27 | re-exports `DistLLMClient` |

## Notes / duplication

- **Two clients named `DistLLMClient`**: `distllm.cli.client` (internal, 188 LOC) shadows the public `distllm.client.client` (423 LOC) and `distllm.sdk.client` (1082 LOC) — a known naming footgun across three namespaces.
- Onboarding/install overlap: `onboard.py`, `setup_wizard.py`, `setup.py`, `install.py` all cover first-run flows — consolidation candidates.
- Model-size/VRAM estimation duplicated in `deploy.py`, `cost_avoid.py`, `tune.py` instead of `config/model_heuristics.py`.

## Depends on
`config` ← only `pydantic`/`yaml`/`loguru`; `cli` ← `config`, `client`, `api` (REST), `core` (cluster/quota); `client` ← `httpx` → [[03 API Server]].
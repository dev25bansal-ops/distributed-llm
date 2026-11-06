---
tags:
  - meta
  - root
  - index
aliases:
  - DistLLM
  - Project Overview
---
# DistLLM — Distributed LLM Inference Engine

**Version:** 0.4.x | **Python:** >= 3.10 | **License:** Apache-2.0 | **Source:** `src/distllm/` — **785 .py files / 225,119 LOC** | **Tests:** `tests/` — **788 files / 152,219 LOC**

> This note is the **landing hub** of the vault. Every source folder of the repository is indexed here and in [[00 Module Index]] — nothing is left unmapped. Use the graph view; the dependency/build folders are excluded so only real project notes appear (see [[#Vault layout & conventions]]).

## What It Is

Pool GPUs across ordinary machines — RTX 4090s, laptops, cloud spots — over plain networking to run LLMs larger than any single box can hold. The engine provides **pipeline / tensor / expert parallelism**, **continuous batching**, **speculative decoding**, **distributed KV cache with prefix sharing**, **cross-cluster federation**, and a full **commercial serving layer** (multi-tenant billing, SSO/RBAC, quotas, SLOs, observability) fronted by an **OpenAI-compatible REST + gRPC API**.

## Architecture

```mermaid
graph TD
    SDK[SDKs: python/go/js/rust + OpenAI compat] --> API[API Server · src/distllm/api]
    Tauri[Tauri+Svelte desktop] -->|HTTP/WS| API
    VSCode[VS Code extension] -->|HTTP| API
    CLI[CLI "distllm"] --> API
    CLI --> Coord
    API --> Coord[Coordinator / src/distllm/core]
    API --> Plugins[Hook plugins / src/distllm/plugins]
    API --> SSO[SSO: SAML/OIDC/OAuth2]
    Coord --> Sched[Batch scheduler + advanced_scheduling]
    Coord --> Engine[InferenceEngine]
    Coord --> KV[Distributed KV cache + prefix cache]
    Coord --> Spec[Speculative decoding family]
    Coord --> Meta[Model/router/cost/marketplace]
    Engine --> Partition[Auto-partitioner / src/distllm/dist/partition]
    Partition --> Worker[(Worker Node / gRPC)] --> GPU[GPU]
    Worker --> Transport[NCCL/gRPC/QUIC/WebRTC/ICE]
    Feeds[cloud / observability / security]

    subgraph Distributed layer
      Pipe[Pipeline orchestration]
      P2P[P2P: gossip / DHT / federation]
      Disagg[Prefill/decode disagg + KV transfer]
      Chaos[Chaos + byzantine + raft]
    end
```

## Module map (where to start)

| MOC | Covers (packages) | Aggregate | Status |
|-----|-------------------|-----------|--------|
| [[01 Core Engine]] | `src/distllm/core/**` | 274 files / ~75K LOC | ✅ rebuilt |
| [[02 Distributed Layer]] | `src/distllm/dist/**` | 194 files / ~71K LOC | ✅ rebuilt |
| [[03 API Server]] | `src/distllm/api/**` | 88 files / ~24K LOC | ✅ rebuilt |
| [[04 CLI & Config]] | `cli/` + `config/` + `client/` | 70 files / ~13K LOC | ✅ rebuilt |
| [[05 Backends & Models]] | `backends/` + `models/` | 29 files / ~8.6K LOC | ✅ rebuilt |
| [[06 Security]] | `security/` + `compliance/` + API auth/SSO + moderation | ~18+ files / ~4.7K+ LOC | ✅ rebuilt |
| [[07 Integrations]] | `integrations/**` + `sdk/**` + `prompts/` + `plugins/` | ~274 files / ~39K LOC | ✅ rebuilt |
| [[08 Frontends]] | `tauri/` + `website/` + `extensions/` + `apps/` + `dashboard|ui` | ~267 files (excl. node_modules) | ✅ rebuilt |
| [[09 Infrastructure]] | `deploy/` + `.github/` + Docker + packaging + `proto/` + `scripts/` + `benchmarks/` | ~144 files | ✅ rebuilt |
| [[10 Tests]] | `tests/**` | 788 files / 152,219 LOC | ✅ rebuilt |
| [[11 Platform Services]] | `observability/` + `health/` + `cache/` + `cloud/` + `errors/` + `utils/` + `terraform/` + `verification/` + `dashboard` | ~61 files / ~13K LOC | ✅ rebuilt |
| [[00 Module Index]] | **Complete index of every folder** (guarantee nothing is missed) | — | ✅ |

<!-- recounted 2026-11-06 by hygiene pass -->

## Aggregate repository statistics

| Tree | Files | Notes |
|------|-------|-------|
| `src/distllm` | 785 .py | 225,119 LOC |
| `tests` | 788 .py | 152,219 LOC |
| `integrations` | 168 | 24 framework adapters + IaC (LangChain, LlamaIndex, CrewAI, K8s operator, TF provider, …) |
| `sdk` | 61 | multi-language: python + go + js + rust + openapi |
| `docs` | 99 | this vault + reference docs (98 .md) |
| `apps` (removed 2026-11-06 hygiene pass) | 0 | chat / RAG / multi-agent demo apps (deleted) |
| `examples` | 20 | drop-in integration scripts + notebooks |
| `benchmarks` | 8 | throughput/latency/scaling harness + saved results |
| `scripts` | 23 | installer, Windows launchers, security audit, CI ratchet gates |
| `extensions` | 17 | VS Code extension (excl. generated `out/`) |
| `website` | 137 | static marketing/product site |
| `tauri` | 92 | desktop app (`node_modules/`/`build/` excluded) |
| `deploy` / `proto` | 69 / 1 | k8s/helm/terraform/kustomize + gRPC contract |

<!-- recounted 2026-11-06 by hygiene pass -->

## Guide to the source packages (src/distllm)

see [[00 Module Index]] for the **complete per-folder index** with one-line purposes and links to the covering note. Highlights:

| Dir | #files | LOC | What it is |
|-----|-------|-----|-----------|
| `core/` | 274 | 75,171 | Heart: Coordinator, batch scheduler, KV cache, speculative decoders, routers, cost/billing, HA, plugins, multimodal |
| `dist/` | 194 | 71,033 | Distributed execution: pipeline, p2p/gossip, partition auto-partitioner, topology, federation, chaos/byzantine, NAT/ICE |
| `api/` | 88 | 23,851 | FastAPI OpenAI-compatible server: routes, middleware, SSO, authz/OPA, WAF, rate-limit |
| `cli/` `config/` `client/` | 70 | 12,850 | Typer CLI (`distllm`), pydantic-settings config, HTTP SDK |
| `observability/` | 20 | 11,755 | Prometheus/OTel/Loguru, SLO config, capacity planning, incident ops |
| `backends/` `models/` | 29 | 8,641 | Inference adapters (vLLM/llama.cpp/Triton/NIM/…) + model partition/adapter/RoPE |
| `security/` `compliance/` | 18 | 4,656 | E2E encrypt, watermark, content moderation, attestation, compliance evidence |
| `cloud/` | 7 | 2,455 | AWS/GCP/Azure pricing + spot orchestrator (RunPod/Vast/Salad) |
| `plugins/` `prompts/` | 26 | 4,897 | Hook plugin system + prompt library/exchange |
| `integrations/` `sdk/` | 19 | 4,072 | in-tree MLflow/Spark/WandB + public python SDK |
| `errors/` `utils/` | 8 | 1,140 | error hierarchy/retry + GBNF/lazy-import helpers |
| `cache/` | 3 | 1,009 | CRDT cross-cluster prefix cache |
| `verification/` | 6 | 1,541 | correctness comparator, hash registry |
| `dashboard/` `health/` `ui/` | 10 | 1,056 | runtime health probes + web dashboards |

<!-- recounted 2026-11-06 by hygiene pass -->

## Cross-cutting themes

- **Compute-shape everything**: GPU pooling → pipeline & KV-cache placement → per-layer quantization (`autoq`, `partition/*`) → cost (`cost_tracker`, `arbitrage`).
- **Spec & cache are first-class**: `speculative_decoder.py`, `dist/proactive_cache`, KV quant, shared-prefix.
- **Multi-tenant commercial layer**: billing/quota/usage-meter, RBAC/SSO, marketplace/DaaS/edge.
- **Defense in depth**: SSRF/IP-pinning, prompt-injection, content moderation, E2E, CRI secrets, key rotation, byzantine/raft, chaos test.

## Recent work (git log highlights)

- **[Strategy 2026-08-08]** Full-repo read → [[_Strategic Recommendations 2026-08-08]] (opportunities, verified issues/roadmap, enhancements, advanced features, testing) — see also the rebuilt [[00 Module Index]] and [[11 Platform Services]].
- `E11` SLA latency tiers per hardware class (provisional) + formatter + regression test — latest commit.
- Comprehensive `dist/` layer audit: fixes, tests, docs.
- Wired 4 new scheduling modules into coordinator production paths.
- Refactors to break `Coordinator`/`BatchScheduler`/`kv_cache.py` monoliths (decomposition refactors) and introduce `utils/lazy_imports.py`.

## Vault layout & conventions

- **Vault root** = the repository root. Notes live in `docs/_map/`. This is an Obsidian vault opened on the repo root.
- **Graph & search** are filtered by `userIgnoreFilters` in `.obsidian/app.json` to exclude `node_modules/`, `.venv*/`, `build/`, `dist/`, `.git/`, caches, `certs/`, and the tooling internals — so the graph shows only real project content.
- Cross-links use `[[Note]]` wikilinks (Obsidian resolves by note name), so the graph lights up from these MOCs.
- **Stray vault stubs (removed):** `distllm/` and `distributed-llm/` at repo root *were* accidental empty Obsidian starter vaults (0-byte notes + default config); they were deleted in the 2026-11-06 hygiene pass and no longer exist at the repo root.

### Navigation

`_Project Overview` (you are here) → [[00 Module Index]] → the [[01 Core Engine]] … [[11 Platform Services]] map notes → the `Core Audit*` notes, the [[Exhaustive Audit 2026-08-11]] full-repo read, the [[Action Plan 2026-08-11]] prioritized roadmap, and `docs/*.md` references.
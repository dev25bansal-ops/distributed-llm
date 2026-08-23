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

**Version:** 0.4.x | **Python:** >= 3.10 | **License:** Apache-2.0 | **Source:** `src/distllm/` — **793 .py files / 222,129 LOC** | **Tests:** `tests/` — **812 files / 228,741 LOC**

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
| [[01 Core Engine]] | `src/distllm/core/**` | 296 files / ~82K LOC | ✅ rebuilt |
| [[02 Distributed Layer]] | `src/distllm/dist/**` | 193 files / ~74K LOC | ✅ rebuilt |
| [[03 API Server]] | `src/distllm/api/**` | 89 files / ~22K LOC | ✅ rebuilt |
| [[04 CLI & Config]] | `cli/` + `config/` + `client/` | 64 files / ~11K LOC | ✅ rebuilt |
| [[05 Backends & Models]] | `backends/` + `models/` | 28 files / ~7.5K LOC | ✅ rebuilt |
| [[06 Security]] | `security/` + `compliance/` + API auth/SSO + moderation | ~18+ files / ~4.3K+ LOC | ✅ rebuilt |
| [[07 Integrations]] | `integrations/**` + `sdk/**` + `prompts/` + `plugins/` | ~300 files / ~45K LOC | ✅ rebuilt |
| [[08 Frontends]] | `tauri/` + `website/` + `extensions/` + `apps/` + `dashboard|ui` | ~270 files (excl. node_modules) | ✅ rebuilt |
| [[09 Infrastructure]] | `deploy/` + `.github/` + Docker + packaging + `proto/` + `scripts/` + `benchmarks/` | ~480 files | ✅ rebuilt |
| [[10 Tests]] | `tests/**` | 812 files / 228,741 LOC | ✅ rebuilt |
| [[11 Platform Services]] | `observability/` + `health/` + `cache/` + `cloud/` + `errors/` + `utils/` + `terraform/` + `verification/` + `dashboard` | ~60 files / ~13K LOC | ✅ rebuilt |
| [[00 Module Index]] | **Complete index of every folder** (guarantee nothing is missed) | — | ✅ |

## Aggregate repository statistics

| Tree | Files | Notes |
|------|-------|-------|
| `src/distllm` | 793 .py | 222,129 LOC |
| `tests` | 812 .py | 228,741 LOC |
| `integrations` | 174 | 28 framework adapters + IaC (LangChain, LlamaIndex, CrewAI, K8s operator, TF provider, …) |
| `sdk` | 58 | multi-language: python + go + js + rust + openapi |
| `docs` | 70 .md | this vault + reference docs |
| `apps` | 12 | chat / RAG / multi-agent demo apps |
| `examples` | 14 | drop-in integration scripts + notebooks |
| `benchmarks` | 22 | throughput/latency/scaling harness + saved results |
| `scripts` | 20 | installer, Windows launchers, security audit, CI ratchet gates |
| `extensions` | 31 | VS Code extension (excl. generated `out/`) |
| `website` | 145 | static marketing/product site |
| `tauri` | ~80 src+config | desktop app (`node_modules/`/`build/` excluded) |
| `deploy` / `proto` | 53 / 1 | k8s/helm/terraform/kustomize + gRPC contract |

## Guide to the source packages (src/distllm)

see [[00 Module Index]] for the **complete per-folder index** with one-line purposes and links to the covering note. Highlights:

| Dir | #files | LOC | What it is |
|-----|-------|-----|-----------|
| `core/` | 296 | 81,744 | Heart: Coordinator, batch scheduler, KV cache, speculative decoders, routers, cost/billing, HA, plugins, multimodal |
| `dist/` | 193 | 67,348 | Distributed execution: pipeline, p2p/gossip, partition auto-partitioner, topology, federation, chaos/byzantine, NAT/ICE |
| `api/` | 89 | 21,832 | FastAPI OpenAI-compatible server: routes, middleware, SSO, authz/OPA, WAF, rate-limit |
| `cli/` `config/` `client/` | 64 | 11,041 | Typer CLI (`distllm`), pydantic-settings config, HTTP SDK |
| `observability/` | 20 | 11,749 | Prometheus/OTel/Loguru, SLO config, capacity planning, incident ops |
| `backends/` `models/` | 28 | 7,544 | Inference adapters (vLLM/llama.cpp/Triton/NIM/…) + model partition/adapter/RoPE |
| `security/` `compliance/` | 18 | 4,336 | E2E encrypt, watermark, content moderation, attestation, compliance evidence |
| `cloud/` | 7 | 2,427 | AWS/GCP/Azure pricing + spot orchestrator (RunPod/Vast/Salad) |
| `plugins/` `prompts/` | 26 | 4,799 | Hook plugin system + prompt library/exchange |
| `integrations/` `sdk/` | 19 | 3,966 | in-tree MLflow/Spark/WandB + public python SDK |
| `errors/` `utils/` | 8 | 1,076 | error hierarchy/retry + GBNF/lazy-import helpers |
| `cache/` | 3 | 1,009 | CRDT cross-cluster prefix cache |
| `verification/` | 6 | 1,541 | correctness comparator, hash registry |
| `dashboard/` `health/` `ui/` | 8 | 637 | runtime health probes + web dashboards |

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
- **Stray vault stubs:** `distllm/` and `distributed-llm/` at repo root are accidental empty Obsidian starter vaults (0-byte notes + default config) — not part of the real vault. Safe to delete.

### Navigation

`_Project Overview` (you are here) → [[00 Module Index]] → the [[01 Core Engine]] … [[11 Platform Services]] map notes → the `Core Audit*` notes, the [[Exhaustive Audit 2026-08-11]] full-repo read, the [[Action Plan 2026-08-11]] prioritized roadmap, and `docs/*.md` references.
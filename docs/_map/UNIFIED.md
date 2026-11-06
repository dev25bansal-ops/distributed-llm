# DistLLM — Unified Project Documentation

**Last consolidated:** 2026-11-06 · **Scope:** this file is the single entry point for everything about the project. Every other doc is linked, not duplicated. When you update code, update the section that owns it.

> **Rule:** one fact lives in exactly one place. This page holds identity + navigation. Details live in the canonical docs linked below. If two docs disagree, this page's link table is the tiebreaker and the loser gets fixed or deleted.

---

## 1. What it is

DistLLM pools GPUs across ordinary machines (RTX 4090s, laptops, cloud spots) over plain networking to run LLMs larger than any single box can hold.

- **Engine:** pipeline / tensor / expert parallelism, continuous batching, speculative decoding, distributed KV cache with prefix sharing, cross-cluster federation.
- **Commercial layer:** multi-tenant billing, SSO/RBAC, quotas, SLOs, observability.
- **API:** OpenAI-compatible REST + gRPC.
- **License:** Apache-2.0 · **Python:** >= 3.10 · **Version:** 0.4.x

## 2. Repository map (verified 2026-11-06)

| Tree | Contents | Canonical doc |
|---|---|---|
| `src/distllm/` | The engine (799 .py / ~272K LOC) | [[#3. Source packages]] + `docs/_map/00 Module Index.md` |
| `tests/` | 946 .py / ~259K LOC, 30+ suites | `docs/_map/10 Tests.md` |
| `sdk/` | Multi-language SDK (python/go/js/rust/openapi) | `docs/SDK_REFERENCE.md` |
| `integrations/` | Framework adapters + K8s operator + TF provider | `docs/_map/07 Integrations.md` |
| `website/` | **Canonical** marketing/product site (CI deploys this one) | `README.md` |
| `website-astro/` | Duplicate site (unreferenced) — see hygiene list | — |
| `tauri/` | Desktop app | `docs/_map/08 Frontends.md` |
| `extensions/` | VS Code extension | `docs/_map/08 Frontends.md` |
| `dashboards/`, `src/distllm/{dashboard,ui}/` | Web dashboards | `docs/_map/08 Frontends.md` |
| `deploy/` | Helm/K8s/Terraform/Kustomize | `DEPLOYMENT.md` |
| `benchmarks/`, `scripts/` | Perf harness, installer, CI gates | `INFRA_CLAUDE.md` |
| `docs/` | All documentation | [[#9. Doc index]] |
| `.github/` | CI workflows, dependabot | `INFRA_CLAUDE.md` |

## 3. Source packages (`src/distllm/`)

| Package | What it is |
|---|---|
| `core/` | Coordinator, batch scheduler, KV cache, speculative decoders, routers, cost/billing, HA, plugins, multimodal |
| `dist/` | Distributed execution: pipeline, gossip/DHT, auto-partitioner, topology, federation, chaos/byzantine, NAT/ICE |
| `api/` | FastAPI OpenAI-compatible server: routes, middleware, SSO, authz/OPA, WAF, rate limit |
| `backends/` `models/` | Inference adapters (vLLM/llama.cpp/Triton/NIM/…) + model partition/adapter/RoPE |
| `cli/` `config/` `client/` | Typer CLI (`distllm`), pydantic-settings config, HTTP SDK |
| `observability/` | Prometheus/OTel/Loguru, SLO config, capacity planning, incident ops |
| `security/` `compliance/` | E2E encrypt, watermark, moderation, attestation, compliance evidence |
| `cloud/` | AWS/GCP/Azure pricing + spot orchestrator (RunPod/Vast/Salad) |
| `plugins/` `prompts/` | Hook plugin system + prompt library/exchange |
| `cache/` | CRDT cross-cluster prefix cache |
| `sdk/` `integrations/` (in-tree) | Public python SDK; MLflow/Spark/WandB adapters |
| `errors/` `utils/` `verification/` `health/` `dashboard/` `ui/` `terraform/` `operator/` | Error hierarchy/retry, GBNF/lazy imports, correctness comparator, probes, dashboards, IaC, K8s operator |

Full per-folder index: `docs/_map/00 Module Index.md`.

## 4. Quick start

- **Docs path:** `docs/QUICKSTART.md` (Docker + Compose), `SETUP_2_LAPTOPS.md` (2-node cluster), `DEPLOYMENT.md` (prod), `docs/air-gapped-deployment.md` (offline).
- **Install:** `install.sh` (CUDA-detecting one-command installer).
- **Develop:** see `CONTRIBUTING.md`; per-area ownership in the `*-CLAUDE.md` files at repo root and inside `src/distllm/*/CLAUDE.md`.

## 5. Configuration

`docs/CONFIG_REFERENCE.md` is the complete option reference. Env vars are read via `distllm.config.settings` (`DistLLMSettings`); `api/api_settings.py` is dead (unimported) — do not edit it for config changes.

## 6. API surface

`docs/api.md` (OpenAI-compatible), `docs/api-versioning.md` (versioning), `docs/API_CHANGELOG.md` (breaking changes), `docs/SDK_REFERENCE.md` (SDKs).

## 7. Operations

`docs/OPS_RUNBOOK.md` (SOPs), `docs/TROUBLESHOOTING.md` (diagnostics), `docs/observability-metrics.md` (metrics reference), `docs/SCALING.md`, `docs/PERFORMANCE_TUNING.md`, `docs/load-testing.md`, `docs/RELEASE_POLICY.md`.

## 8. Security & compliance

`SECURITY.md` (policy), `docs/threat-model.md`, `docs/SECURITY_HARDENING.md`, `docs/GDPR.md`, `docs/EXPORT_CONTROLS.md`, `docs/air-gapped-deployment.md`.

## 9. Doc index (every doc, one line, owner)

_Filled by the hygiene pass — see `docs/_map/` for the audit vault. Historical audits live in `docs/_map/` (Obsidian vault) and `docs/_audit/`; they are reference material, not living docs._

## 10. Hygiene log

Every deletion/move with its evidence. Format: `YYYY-MM-DD — path — verdict — evidence`.

**2026-11-06 pass (this session, evidence per item):**
- `.distllm/` (753 files) — DELETE: runtime data (batches/uploads/images/SQLite) written by `get_data_dir()` default (persistent_store.py:303); zero fixture reads repo-wide; now gitignored.
- `_quarantine/`, `review_findings.json` — DELETE: zero importers; only test *names* reference "quarantine".
- Root scratch scripts (`run_all.py`, `run_all_tests.py`, `run_tests_v2.py`, `run_checks.sh`, `run_check.bat`, `run_tests.bat`, `run_tests.ps1`, `runner.sh`, `step1.py`, `__check_syntax.py`) — DELETE: referenced by NOTHING; hardcoded `D:/distributed-llm` paths break portability.
- `.agent/`, `.lgraph/`, `.orchids/` — DELETE: vendor/scratch tooling metadata, no references.
- `website-astro/` — DELETE: duplicate site; CI deploys `website/` only (deploy-website.yml:8); FRONTENDS_CLAUDE.md owns `website/`.
- `website/index2.html`, `website/index-fixed.html` — DELETE: unreferenced editor stash snapshots.
- `dashboards/distllm-grafana.json` — DELETE: queries metrics that don't exist (observability-metrics.md:173 says legacy); `distllm-overview.json` is the reference dashboard.
- `.gitignore` — FIX: added `.distllm/` + `.usage.db*` rules (classifier found the gap; `git check-ignore` confirmed).

_The 8-agent classifier pass is still landing; remaining verdicts (dead src modules, doc fixes, SDK/integration dupes) append here._

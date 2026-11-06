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

## 9. Doc index (every living doc, one line, owner)

_Living docs only. Historical audit snapshots in `docs/_map/` carry a HISTORICAL SNAPSHOT banner — read them as records, not references._

| Doc | What it owns |
|---|---|
| `README.md` | Project identity, quick links, badges |
| `CHANGELOG.md` | Release notes (Keep a Changelog format) |
| `CONTRIBUTING.md` | Dev setup, PR process, area ownership (`*-CLAUDE.md`) |
| `DEPLOYMENT.md` | Local/Docker/Compose/K8s deployment |
| `SECURITY.md` | Security policy, supported versions |
| `SETUP_2_LAPTOPS.md` | 2-node cluster guide (sanitize personal IPs before reuse) |
| `TASKS.md` | Priority-ordered engineering task list |
| `docs/QUICKSTART.md` | Fastest path to a running cluster |
| `docs/ARCHITECTURE.md` + `docs/ARCHITECTURE_DIAGRAMS.md` | System design + mermaid diagrams |
| `docs/api.md`, `docs/api-versioning.md`, `docs/API_CHANGELOG.md` | API surface, versioning policy, breaking changes |
| `docs/CONFIG_REFERENCE.md` | Every config option (authoritative) |
| `docs/SDK_REFERENCE.md`, `docs/BACKEND_PLUGIN_GUIDE.md`, `docs/PLUGIN_DEVELOPMENT.md` | SDK + plugin extension guides |
| `docs/OPS_RUNBOOK.md`, `docs/TROUBLESHOOTING.md`, `docs/observability-metrics.md` | Operations, diagnostics, metrics dictionary |
| `docs/SCALING.md`, `docs/PERFORMANCE_TUNING.md`, `docs/PERFORMANCE_COMPARISON.md`, `docs/load-testing.md`, `docs/benchmark-results.md`, `docs/BENCHMARKS.md` | Perf evidence + how to reproduce |
| `docs/SECURITY_HARDENING.md`, `docs/threat-model.md`, `docs/GDPR.md`, `docs/EXPORT_CONTROLS.md`, `docs/air-gapped-deployment.md` | Security/compliance evidence |
| `docs/MIGRATIONS_README.md`, `docs/MIGRATION_*.md` | Migration guides (OpenAI/vLLM/Ollama/v0.3→v0.4) |
| `docs/MODEL_COMPATIBILITY.md` | Model × backend matrix |
| `docs/SCALING.md`, `docs/SLA_TIERS.md`, `docs/RELEASE_POLICY.md` | Ops policy |
| `docs/MARKETING.md`, `docs/FINANCIALS.md`, `docs/CLOUD_MARKETPLACE.md`, `docs/YC_STARTUP_PACKAGE.md`, `docs/competitive-analysis.md`, `docs/demo-script.md`, `docs/ACADEMIC_PARTNERSHIPS.md`, `docs/OPEN_SOURCE_CONTRIBUTIONS.md`, `docs/TERMS.md`, `docs/self-hosted.md`, `docs/REMOTE_CONNECTIONS.md`, `docs/EXAMPLES_APPS_NOTEBOOKS.md`, `docs/FOCUS-AREAS.md`, `docs/HYBRID_CONTROL_PLANE.md` | Go-to-market, legal, partnerships (historical business docs — read-only) |
| `docs/_map/00–11` | Obsidian MOC vault: per-folder module maps (recounted 2026-11-06) |
| `docs/_map/UNIFIED.md` | This file — single entry point |
| `docs/adr/0001–0005` | Architecture decision records |
| `docs/_audit/` | Raw exhaustive-audit working files (historical) |

**Deleted this pass:** `docs/EXAMPLES_APPS_NOTEBOOKS.md` dead `../apps/README.md` link (apps/ gone), `RELEASE_POLICY.md` phantom `MIGRATION_vX_to_vY.md` (now points at the real template). `TASKS.md:107` names an ADR that does not exist yet — that is a TODO, not a dead link; left as-is.

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
- `integrations/` DELETE: README-only stubs `genkit/`, `openwebui/`, `portkey/`, `spring-ai/` (zero code, zero refs outside audit docs); stale loose duplicates `mlflow_tracking.py`, `spark_connector.py`, `wandb_tracking.py` (live copies in `src/distllm/integrations/` + `integrations/wandb/`); `ANALYSIS.md` (misdescribes tree).
- `sdk/` DELETE: `dist/distllm_sdk-0.5.0.tar.gz` (stale vs pyproject 1.0.0), `go/_smoke/` (orphan, hardcoded demo key). FIX: `sdk/dist/` + `sdk/js/dist/` gitignored (npm main points at generated output; rebuild in CI).
- `benchmarks/` DELETE: `competitive_benchmark.py` (unreferenced twin of `run_competitive.py`), `compare.py`, `cost_routing_benchmark.py`, `Dockerfile` (COPY dist/ — no dist/ exists).
- `tests/benchmarks/` DELETE: plural dupe of CI-wired `tests/benchmark/` (Makefile:44, benchmark-regression.yml:44).
- `dashboards/distllm-grafana.json` DELETE: queries nonexistent metrics (observability-metrics.md:173 marks legacy); `distllm-overview.json` is the reference.
- `website/index2.html`, `website/index-fixed.html` DELETE: unreferenced editor stashes.
- `framework_adapters.py:355` FIX: removed dead `"example": "examples/agno_example.py"` (file never existed); `list_frameworks` now `info.get("example")`.
- `examples/notebooks/{distllm_kaggle_worker,distllm_lightning_worker,tinylm_colab_worker}.ipynb` FIX: hardcoded `D:\distributed-llm\examples\...` → relative `python distllm_laptop_coordinator.py`.
- `Makefile` + `.github/workflows/ci.yml` FIX: helm paths `deploy/helm/` → `deploy/helm/distllm-operator/` (Chart.yaml lives one level deeper; CI helm job was failing).
- `docs/RELEASE_POLICY.md` FIX: phantom `MIGRATION_vX_to_vY.md` → template pointer. `docs/EXAMPLES_APPS_NOTEBOOKS.md` FIX: dead `../apps/README.md` link (apps/ deleted).
- `docs/_map/*` FIX: 26 historical audit snapshots stamped HISTORICAL SNAPSHOT banner (kept in place — moving breaks Obsidian `[[wikilinks]]`); living MOCs recounted by dedicated agent.
- `src/distllm/core/framework_adapters.py` FIX: see above.

**Kept despite looking like junk (verified references):** `requirements.lock` (Makefile+Dockerfile), `config.yaml`/`install.sh`/`docker-entrypoint.sh` (referenced), `integrations/kubernetes/` (helm-publish.yml:45 packages it), `sdk/js/dist/` (npm main), `deploy/ray/`+`deploy/operator/controller.py`+`deploy/webhook/` (REVIEW — orphan deploy surfaces, decided KEEP pending product direction), `src/distllm/ui/` (REVIEW — redundant with dashboard but tested+packaged).

**Open follow-ups (REVIEW items, not deleteable unilaterally):**
- `src/distllm/dashboard/static/` vs `static_v2/` — v1 still served by server.py; needs one product decision, then delete v1.
- `src/distllm/dashboard/static_v2/verification_report.html` — one-off snapshot in served static dir.
- `tauri/COMPREHENSIVE_ANALYSIS.md` — stale analysis artifact inside product code.
- `src/distllm/` dead code (79 modules, commit 6cec124): AST import graph + repo-wide grep, zero importers each. api: api_settings, api_versioning, auth/store, background_tasks, grpc_bridge, token_estimator, ws_chat, server_routes_api, server_routes_dashboard, services/{chat,completion,embedding}_service, routes/{assistants,experiments}. cli: 12 inline-duplicated command modules. prompts: 11 category modules (inline in library.py). observability: 8 config orphans. plugins/{airflow,kubeflow,registry}, sdk/observability, utils/lazy_imports, terraform/__init__, models/{_trust_remote,adapter_router}, config/model_heuristics. core: 25 orphans + paired tests (cache_*, hierarchical_digest, cluster_registry, cost_comparison, distributed_distillation, feature_flags, gossip_cache_bridge, gpu_power_manager, hardware_plugin_registry, hibernation_manager, kv_backup, kv_cache_compressor, micro_batch_scheduler, model_selector, model_sizing, persistent_store, pipeline_overlap, preemptible_scheduler, request_auditor, types, webgpu_manager). bandit-baseline −180 entries, real_import_baseline −1.
- **Kept despite zero importers**: deprecated re-export shims (latency_tracker, llamacpp_backend, pipeline_orchestrator, predictive_cache, straggler_detector, vllm_backend) — public API compat per API_CHANGELOG; redis_rate_limiter (3 test files load by path); server_routes_dashboard sibling under review; evaluation_harness (kept, fixed to raise NotImplementedError instead of fabricating accuracy); deploy/{ray,operator,webhook} (self-contained configs/manifests); integrations/one-api (standalone pip pkg); src/distllm/ui (tested, product decision pending).
- `docs/CLOUD_MARKETPLACE.md` DELETE: cited deploy/{aws,gcp,azure} paths all absent; vaporware doc.
- `docs/GDPR.md`, `api-versioning.md`, `SECURITY_HARDENING.md`, `OPS_RUNBOOK.md`, `MIGRATION_v0.3_to_v0.4.md`, `MIGRATION_OPENAI.md`, `QUICKSTART.md`, `air-gapped-deployment.md`, `PERFORMANCE_TUNING.md`, `SCALING.md`, `TROUBLESHOOTING.md`, `ARCHITECTURE.md`, `MODEL_COMPATIBILITY.md`, `BENCHMARKS.md`, `FOCUS-AREAS.md`, `EXAMPLES_APPS_NOTEBOOKS.md`, `OPEN_SOURCE_CONTRIBUTIONS.md`, `YC_STARTUP_PACKAGE.md`, `REMOTE_CONNECTIONS.md`, `ACADEMIC_PARTNERSHIPS.md`, `COMPREHENSIVE_ANALYSIS.md`, `SDK_REFERENCE.md`, `PLUGIN_DEVELOPMENT.md`, `api.md`, `API_CHANGELOG.md` — FIX: dead endpoints/paths/commands/env-var fictions corrected against src (each fix cites grep evidence); `_map/03 API Server.md` api_versioning marked deleted.
- Living MOCs `docs/_map/00–11` + `_Project Overview.md` — FIX: aggregate counts deterministically recounted (src 785 files/225,119 LOC pre-79-deletion; recount against post-deletion tree pending final pass); `<!-- recounted 2026-11-06 by hygiene pass -->` markers; apps/ row marked removed; historical files byte-identical.
- Permanently-skipped tests (module-level `pytest.skip(allow_module_level=True)`) — list in classifier output; fix env or delete file-by-file.
- `docs/_map/08 Frontends.md` documents a gone `apps/` tree (recount agent will fix).

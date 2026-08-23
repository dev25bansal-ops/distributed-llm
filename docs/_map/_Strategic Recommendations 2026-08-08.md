---
tags:
  - meta
  - strategy
  - roadmap
  - audit
aliases:
  - Strategic Recommendations
date: 2026-08-08
---
# Strategic & Technical Recommendations — 2026-08-08

> Full-repository strategic and engineering assessment. Produced after a line-by-line read of every source file (`src/distllm` = 793 files / 222,129 LOC; `tests` = 812 files / 228,741 LOC; plus `integrations`, `sdk`, `frontends`, `deploy`, `.github`). Every "broken / dead / unwired" claim below was confirmed against the current working tree.
>
> Scope note: findings are grounded in the vault MOCs — [[01 Core Engine]], [[02 Distributed Layer]], [[03 API Server]], [[04 CLI & Config]], [[05 Backends & Models]], [[06 Security]], [[07 Integrations]], [[08 Frontends]], [[09 Infrastructure]], [[10 Tests]], [[11 Platform Services]], [[00 Module Index]].

---

# 1 · Project Analysis & Strategic Opportunities

## 1.1 Where this project genuinely differentiates (strengthen these)

1. **Consumer-grade heterogeneous GPU pooling over the open internet.** NAT/ICE/STUN/TURN traversal (`dist/ice_transport.py` 2481 LOC, `dist/nat.py`, `dist/webrtc.py`), Edge-to-Cloud continuum (`dist/edge_cloud.py`), browser GPU contribution (`core/wisp_wasm.py`, `backends/webgpu_backend.py`), and multi-tenant DaaS (`dist/daas/`). This is the **rarest capability in the market** — most rivals assume controlled DCs. It is the differentiation story.
2. **A full commercial serving layer included out of the box** — multi-tenant billing/quota (`core/usage_meter.py`, `tenant_billing.py`), SSO/RBAC (`api/auth/*`, `authz/opa.py`), marketplace (`dist/marketplace.py`), leaderboard — most OSS competitors lack billing/tenancy.
3. **Speculative-decoding + KV-cache depth is exceptional** — family in `core/speculative*`, WAN token accumulation (`dist/wan_speculative.py`), online self-correcting (`dist/speculative/online_sisd.py`). SGLang-level ambition on a P2P substrate.
4. **Green/energy/carbon-aware routing** (`core/advanced_scheduling/energy.py`, `core/carbon_migration.py`, `core/cross_cloud_router.py`, `core/pareto_optimizer.py`) — a genuine "carbon-aware AI inference" pitch.
5. **Compatibility surface is unusually broad** — OpenAI/Llama/Ollama/TGI compat, 28 framework adapters, multi-language SDKs (python/go/js/rust).
6. **Correctness obsession** — `core/correctness_harness.py` + HMAC certs, `verification/hash_registry.py`, PBT suites. Rare in this space.

## 1.2 Where the market is ahead (gap to close)

- **vLLM / SGLang production maturity**: continuous-batching + PagedAttention throughput at scale, block-manager servicing, structured topology, kernel quality. DistLLM is feature-rich but risks "wide and shallow" — many capabilities are SCAFFOLD/stub or untested at real GPU scale.
- **Ollama/Windows "one-click" UX**: ours still needs `install.sh`/docker/manual multi-node bring-up. Onboarding and the **single-machine path** are the biggest usability wins available.
- **Llama-Stack serving**: integration parity with `llama-stack` APIs would lower adoption friction.
- **RAG/agents assembly** vs ecosystem-native offerings: we provide blocks, not opinionated recipes.

## 1.3 Strategic opportunities (prioritized)

| # | Opportunity | Play | Primary moat | Effort |
|---|-------------|------|--------------|--------|
| S1 | **"Bring-Your-own-GPU over WAN" as the flagship** — productize the P2P/ICE/disaggregation + marketplace into a managed multi-node-in-minutes experience and a public capacity marketplace | Make the demo: launch a 2-node cluster across two homes in <3 min from the CLI; publish a **DistLLM network** for spare-GPU trading. | Hard to replicate (ICE/DHT/reputation), can't be copied quickly | L |
| S2 | **Carbon & power-aware inference as an enterprise wedge** — expose `cross_cloud_router`/`energy`/`power_cap` as an add-on with dashboards + SLO | TCO + sustainability narrative | measurement + policy differentiation | M |
| S3 | **Edge/browser + Wasm serving** — monetize `wisp_wasm`/`webgpu`/`edge_federation` as a low-cost edge tier | Cost/familiarity, new device surface | rare capability | M |
| S4 | **A "switchable inference broker" product** — multi-provider routing (`UnifiedSlaRouter`, `CrossCloudRouter`, `Arbitrage`) surfaced as "put any OpenAI key in, we route best". Covers cost/carbon/latency. | Direct revenue path | routing+measurement | M |
| S5 | **Benchmark-led credibility** — publish reproducible comparisons vs vLLM/SGLang/TGI (already exists in `benchmarks/competitive_benchmark.py`) as marketing + regression gate | Adoption + trust | data | S |

**Measurable leadership goal:** "Best cost-per-token for a distributed cluster + only product that pools home GPUs." Position against Ollama (single-box), vLLM (DC), and Bittensor (crypto) by being **the practical, billing-ready, multi-provider engine**.

---

## 2 · Issues & Required Fixes

> **Resolution status (2026-08-08):** I1, I2, I3, I4 are now **FIXED** (verified + tested). I5–I18 remain open. See the git diff and `src/distllm/api/CLAUDE.md`.

Rating legend: **Severity** — Critical (C) / High (H) / Medium (M) / Low (L). **Effort** in engineer-days (range). **Priority** 1=do now … 4=backlog. **Timeline** = recommended quarter.

### 2.1 Confirmed defects & dead code

| ID | Severity | Effort | Prio | Verify | Issue | Timeline |
|----|------|-----|-----|-----|----------------|----------|
| I1 | C | 0.1d | 1 | ✅ tree-checked | `src/distllm/benchmarks/__init__.py` imports `distllm.benchmarks.cost_comparison`, but **`benchmarks/cost_comparison.py` does not exist** → `import distllm.benchmarks` raises `ModuleNotFoundError`. | Q3 |
| I2 | H | 0.3d | 1 | grep: 0 imports | `api/server_middleware.py` (394 LOC) is **dead** — not imported anywhere; worse, `api/CLAUDE.md` claims it is "(DELETED — dead code removed)" so docs lie. Delete + fix CLAUDE.md. | Q3 |
| I3 | H | 1–2d | 1 | grep | `sso_middleware.py` (626 LOC) defines the SSO middleware + `/v1/auth/*` endpoints but is **not wired into `server.py`** — those endpoints defined but unmounted (only the `sso_auth.py` core + `auth/*` handlers are live). Either mount it or delete + move the useful `/auth/token` handlers. | Q3 |
| I4 | M | 0.2d | 2 | read | **SDK version skew**: `sdk/pyproject.toml` = 1.0.0, `sdk/src/distllm_sdk/__init__.py` = 1.0.0, `sdk/rust/Cargo.toml` = 0.4.0, built wheel `distllm_sdk-0.5.0` → three divergent version strings. Collate to a single released version. | Q3 |
| I5 | M | 0.2d | 2 | read | `sdk` has **two divergent OpenAI-compat clients** (`sdk/src/distllm_sdk/compat/openai_compat.py` newer, `sdk/compat/openai_compat.py` older) plus a generated set. Consolidate to one + one `_base_headers`/retry path. | Q3 |
| I6 | M | 0.2d | 3 | read | `sdk` defines **4 duplicated type ecosystems** (types.py / types_dataclass.py / generated/types.py / compat) — pick one codegen source (openapi) and delete the rest. | Q4 |
| I7 | M | 0.3d | 3 | read | Two parallel **prompt libraries** (`prompts/library.py` live; `prompt_def.py` + 11 category modules dead duplicate, plus `management.py`). Delete the dead path. | Q3 |
| I8 | M | 0.5d | 3 | architect | **Three `DistLLMClient` namespaces** collide: `distllm.cli.client` (188) vs `distllm.client` (423) vs `distllm.sdk.client` (1108). Rename/alias to a single public `distllm.sdk` while keeping back-compat shims. | Q4 |
| I9 | M | 0.5d | 3 | architect | `dist/` STUN/TURN/ICE triplicated across `nat.py`, `webrtc.py`, `ice_transport.py` (doc says webrtc supersedes nat). Remove `nat.py` gen; centralize on `ice_transport`. | Q4 |
| I10 | M | 0.5d | 3 | architect | Redundant execution duplicated: `dist/redundant.py` (older) vs `dist/redundant_executor.py` (production). Consolidate to production. | Q4 |
| I11 | M | 0.2d | 3 | read | **Namespace collision**: `src/distllm/dist/backends/` and `src/distllm/backends/` are distinct packages sharing the name `backends` — easy to import the wrong one. Rename/alias the `dist` variant. | Q4 |
| I12 | L | 0.1d | 4 | read | Dashboard **v1 + v2 duplicated** (`dashboard/static` vs `static_v2`), `dashboard/ui/` empty dir; merge to v2, drop v1. | Q4 |

### 2.2 Under-engineered / stub-heavy areas (the "looks shallow" risk)

Highest credibility risk: many modules are intentionally over-scaffolded (`SCAFFOLD`) rather than production code, yet can read as shipped. Verified markers that need a productize / integrate / remove decision:

- `security/edge_attestation.py`, `security/spiffe.py`, `security/tee.py`, `security/quantum_safe_tls.py` — all **SCAFFOLD** with dev keys on disk ("DEV ONLY"); not production.
- `cloud/worker_agent.py` (E13) — joins a **non-existent control plane**.
- `deploy/operator/controller.py` — `logger.info` only, no k8s API calls.
- `observability/self_healing_config.py::_monitor_loop` and `capacity_planning.py::_collect_snapshot` — placeholder/mocked telemetry.
- `dist/ebpf_transport.py` — software-simulated obs/offload.

**Recommendation:** form an explicit "Production Readiness Decks" — for each of these, decide **productize / integrate / remove**. Document status + owner. Count: ~8 stubs not to be mistaken for delivered capabilities. This is the single largest credibility risk — docs/README may overpromise.

### 2.3 Quality & behavioral risks

| # | Severity | Effort | Prio | Issue | Timeline |
|---|------|-----|-----|-------|----------|
| I13 | H | 1–2d | 2 | **`sso_middleware` unwired** leaves the documented SSO flow unreachable; either finish wiring or trim the docs (see I3). | Q3 |
| I14 | H | 0.5d | 1 | **`I1` (`benchmarks/__init__`) blocks `import distllm`** in any tool that does `importlib` on the whole tree (CI import-all gates may be silently skipping `distllm.benchmarks`). | Q3 |
| I15 | M | 1–2d | 3 | Heavy **stub/fixture reliance** — `api`, `cli`, `core`, `integration`, `verification` packages have large `_stubs.py`/`stubs.py` substitutions; real-GPU/real-cluster coverage is thin at unit level. Fill the GPU matrix via self-hosted runners. | Q4 |
| I16 | M | 0.5d | 2 | **Unused/vacuous code path** in `integrations/spark_connector.py::_call_batch_udf` — `any(attempt<... for attempt in range(...))` always true. | Q3 |
| I17 | M | 0.3d | 2 | `core/plugins/` and `plugins/sandbox.py` pure re-export shims; `mlflow_plugin.py` doesn't subclass `PluginBase`. Clarify seam or remove. | Q4 |
| I18 | M | 1–2d | 3 | `config`/`cli`/`deploy`/`cost_avoid`/`tune` each re-implement model-size/VRAM estimation instead of `config/model_heuristics.py` — single source. | Q4 |

### 2.4 Dependencies & sequencing
Do I1→I2→I14 first (they bite immediately: import breakage + docs+memory). Then I3/I4/I5 (versioning/Auth trust). Then I6–I18 (consolidation). Then the stub productionization sprints. Roughly: **Week 1** = I1,I2,I14; **Weeks 2–3** = I3,I4,I5,I7,I16,I17,I18; **Weeks 4–6** = I6,I8,I9,I10,I11,I12 + scaffold triage.

---

## 3 · Enhancements & Modifications

| Where | What | Why / Benefit | Trade-off / Notes |
|-------|------|----------------|-------------------|
| `api/` | **Single auth-decision point** | Current stack has AuthMiddleware + RBAC + SSO + plugin + quota with logic split (`middleware.py`, `sso_middleware.py`, `authz/opa.py`, `plugins/auth_plugin.py`). Consolidate into one `Authorize(request)` pipeline that returns uid/roles/tenant; every route reads `request.state.authz`. Reduces bypass risk & duplication. | Medium refactor; watch perf. |
| `backends/`+`models/` | Fix `__init__.__all__` inconsistencies (several symbols only reachable via submodule import) | Public API correctness; reduces confusion. | — |
| `config/` | Adopt `Settings` myconfig from a `.distllm` user config + expose `DistLLMSettings.model_validate` everywhere | removes the ~4 duplicated copies of lazy-import and settings-read | |
| `dist/pipeline/` | **productize `pipeline_reconfig.PipelineReconfigurator`** with a CLI/API `replan` command and rollback to last-known-good partition | It exists (1384 LOC) but under-exposed; make reconfig a first-class ops action | needs live-testing |
| `core/` | Wire the `DisaggregatedScheduler`/`DisaggOrchestrator` into a **`prefill/decode` deployment flag** (it's already implemented but not surfaced) | K/V transfer is the headline perf lever | complex |
| `observability/` | **Unify the two health impls** (`health/` pkg vs `observability/health_config.py`) | single HealthCheckService | medium |
| `errors/` | Ensure every raised `DistLLMError` carries `code`/`docs_url` | error trait — already has `troubleshooting_section`; wire to docs portal | low |
| `cli/` | Merge `onboard.py/setup_wizard.py/setup.py/install.py` into **one** `distllm init` flow | remove 4-way overlap (naming footgun); faster first-run | low |
| `plugins/`+`prompts/` | Delete the dead prompt path + unify plugin registry with entry-point discovery | less code, fewer surprises | low |

---

## 4 · Advanced Features (competitively differentiated)

| Feature | Rationale & value | Implementation sketch |
|---------|-------------------|-----------------------|
| **F1 · WebRTC/Wan disaggregated prefill/decode at production behavior** | `WanDisaggOrchestrator` + `IPC transfer` already exist; turning this into a default "placement: disagg" for multi-home clusters is a headline differentiator (low TTFT over WAN). | surface a `--mode disagg` flag; wire autoscaler to prefill/decode pools; SLO-threshold fallback to co-located. |
| **F2 · Online speculative-decoder learning as a knob** | `online_sisd.py` (self-correcting) + `draft_quality_scorer` — make acceptance-rate-adaptive a per-tenant setting | engine + eval-gate |
| **F3 · Carbon/power-metered inference** | Already has `energy.py`, `power_cap.py`, `carbon_migration.py`, `pareto_optimizer`. Expose carbon budget as a per-request header + dashboard; GHG-report ready. | config + observability |
| **F4 · Trusted compute frontier** (the scaffold upside) | `tee.py`/`edge_attestation`/`quantum_safe_tls`/`DP inference` can become a **Confidential-Inference product** (attestation + DP + e2e + k-anon); capitalizes on S1 privacy moat. | high — needs attestation backend |
| **F5 · Real decentralized "markets"** | `dist/marketplace.py` + `core/kv_cache_marketplace.py` + edge_federation + DHT → a live P2P GPU/KV marketplace with a trust ledger (`core/federated_incentives.py`). | product + security |
| **F6 · Correctness-guarantee cert ledger** | `correctness_harness` + `verification/hash_registry` + on-chain/manifest → a "proven-correct" attestation for results | Low |
| **F7 · Cluster **self-healing** with a SLO playbook** | `self_healing_config` + chaos → turn into always-on "resilience SLO" that proves MTTR | ops engineering |
| **F8 · Intent-based topology** | `neural_partition_optimizer` + `DigitalTwin` → user says "this cluster, this model, tslo X"; system suggests an optimal (recursive) partition | medium |

---

## 5 · New Additions (new modules / integrations)

| # | Addition | Value | Effort |
|---|----------|-------|--------|
| N1 | **`DistLLM Cloud` control plane (the E13 spawn)** — a hosted coordinator that picks spot + a `worker_agent` that registers. This is the single biggest untapped asset (the worker join code already exists). | opens a hosted product + adoption | L |
| N2 | **`distllm-server-free` single-binary/`docker compose` one-click "Tiny cluster in a box"** for laptops + a `distllm launch --join` | onboarding UX moonshot (Ollama-parity) | M |
| N3 | **Stripe billing + metered usage webhooks** over `usage_meter`/`metering` to "charge by token" | SaaS revenue | M |
| N4 | **OpenTelemetry → (LangTrace/Langfuse/SignalFx) trace export** | it's infra-native, + AI-observability integrations | low |
| N5 | **A `Streamlit`/`Gradio` demo chat/evals bundle** in apps/ | developer relations | low |
| N6 | **Publish SGLang-** and **Llama-Stack** compat shims | ecosystem parity | low |
| N7 | **Add `Server-Sent-Events` + `OAuth2`+OpenID full OIDC flows to `/v1/auth`** (currently unwired — see I3) | SSO customers | M |
| N8 | **K8s operator GA-ing** (make `deploy/operator/controller.py` real) | enterprise | M |
| N9 | **Billing ledger as DB-backed (Postgres) + `gRPC` for files** | scale | M |
| N10 | **A "cluster doctor" as a SaaS endpoint** — plumbing `cli/doctor.py` behind `/health` detailed | supportOps | low |

---

## 6 · Verification & Testing Strategy

Overall: the suite is **exceptional in breadth** (812 test files / 229K LOC, fuzz + property + mutation + chaos + correctness). The gaps are *depth under real hardware* and *pre-existing skips/work-tree bugs*.

| Area | Recommendation |
|------|----------------|
| **Unit** | Keep per-module `test_<module>.py`; run mutation floor (`tests/mutation/mutate.py`), coverage ≥ 80 gate in CI. |
| **Integration (GPU)** | Migrate the self-hosted `gpu-tests.yml` to cover for **each backend**: torch, vLLM, ONNX, llama.cpp, Triton. Real 2-node `dist/integration` + `tests/distributed/test_real_multi_gpu.py` on NVLink + GbE. |
| **E2E** | Keep `e2e/` (+ multi-node/docker); add **our own** "one-cloud launch" golden-path as an E2E spec that takes ≤2 min. |
| **Performance** | `benchmarks/compare.py` runs on a dedicated GPU bench runner; gate any PR that regresses >2% throughput or +5% TTFT (baseline in `baseline.json`). |
| **Property** | Extend property/PBT suites to all pure modules (router, quant, scheduler); run Hypothesis in CI. |
| **Fuzz** | Keep `tests/fuzz/*` — add a **JSON-schema fuzzer** toward `structured_output`, and a proto fuzzer toward `node_pb2`. |
| **Security** | Add a live `nikto`/`zap`/DAST job; gate on `bandit+detect-secrets+pip-audit` (all in scripts). |
| **Regression** | `tests/regression_high/` = 70 gold gates; `regression_check.py` compares vs baseline and flames from cProfile. |
| **UAT** | Use `apps/chat`, `apps/rag`, `apps/multi_agent`, the four notebooks, and `cli` as the acceptance harness; run a monthly "product E2E prep" check. |
| **Hard** deployment in CI | `deploy/helm`, `kustomize`, `operator` tests + a `docker` smoke; keep `deploy/tests` green. |

**Meta-gap to close:** the "pre-existing worktree test bugs" flagged in the release gates (memory note) — a quarterly `make check` blitz to eliminate hard skips/flakes and to make sure CI has no silent skips / over-used xfails.

---

## 7 · Beyond the categories — additional operating plans

### 7.1 Roadmap (0.4.x → 1.0) & phasing
- **0.4.5** (this quarter): Q3 housekeeping (issues I1–I5), retire the scaffold stubs, publish the "Carbon/Latency" leaderboards, + a live `disagg` flag.
- **0.6.0**: performance baselines made real; GPU CI matrix; multinode E2E.
- **0.7.0**: the "cloud control plane" (N1), marketplace GA, Disagg GA.
- **1.0**: trusted-compute + compliance evidence automation + metered billing GA + publish open benchmarks.

### 7.2 Security operations
- Turn on the **`vendor security` ratchet gates** (bandit + secrets + coverage + flaky) as **blocking** in CI, not advisory.
- **Quarantine the 60+ scaffolds** (2.2). For each, link a ticket + a security owner; no production data claim strips allowed.
- Enable **attestation as default** for multi-node; document the `DISTLLM_LEGACY_CLUSTER_KEY` → SVID migration.

### 7.3 Reliability & SLO
- Define external SLOs (target: 99.9% API availability, 99.0% cluster converge); wire `slo_config.py` burn-rate to alerts.
- Run `recovery_drill.py` on a scheduled chaos cadence to validate MTTR.

### 7.4 Engineering metrics (OKRs)
| Metric | Target Hz |
|--------|-----------|
| PR-to-merge lead time | <3h core |
| Flaky rate | <1% |
| Mutation survival | ≤ baseline |
| Coverage (core) | ≥ 85% |
| Bench regressions | 0 sustained |

### 7.5 Community & adoption
- Public benchmarks + `BENCHMARKS.md` (already there) + a **"Run it in 5 min"** README path.
- Publish the **OpenAI-compat pass-rate** and a `compat/` conformance suite (the single most re-assuring POC signal).
- Keep the **`ACTIVE` downstream packages** green: langchain / llamaindex / crewai (already wired).

### 7.6 Docs / vault
- The vault (`docs/_map`) is already rebuilt — keep MOCs updated on every module death/consolidation (e.g., when I1/I2 are resolved, remove those entries and fix the stale `api/CLAUDE.md`). Add a `_CONTRIBUTING` note.

---

## Appendix — Confirmed "must-fix" shortlist (bite the avoidable)
1. **I1** benchmarks import crash → fix in 1 h.
2. **I2** `server_middleware.py` dead + CLAUDE.md stale (1 h).
3. **I3** SSO middleware unwired (0.5–1 d).
4. **I4–I6** version/type/OpenAI-compat duplication (0.5–1 d).
5. **I13–I18** single-source config / controller / scaffolding triage (2-3 d).

**Escalators**: any of I1 / I2 / I3 / I5 failing a fresh `python -c "import distllm"` or `make check` should halt the minor release.

---

*This document cross-references: [[_Project Overview]], [[00 Module Index]], and each MOC under the "recent work" audit trail.*
---
tags:
  - core
  - audit
  - meta
date: 2026-08-05
---

# DistLLM Core — Comprehensive Audit 2026-08-05

**Scope:** `src/distllm/core/**` — **296 Python files, ~97,315 LOC** (242 top-level + 54 across `advanced_scheduling/`, `dp_inference/`, `evaluation/`, `plugins/`, `scheduler/`, `structured_output/`, `vectorstore/`).
**Method:** Exhaustive line-by-line read of every file by 14 parallel zone readers + 1 strategic reader + 1 testing reader; adversarial verification of Critical/High bug + security findings.
**Result:** **202 findings** across all categories.

## Status update — release blockers fixed (2026-08-05)
The four code release-blockers **C1–C4 are now FIXED with passing regression tests** (`tests/core/test_coordinator_startup_order.py`, `test_kv_cache_quantized_path.py`, `test_structured_output_first_call.py`, `test_async_pipelined_speculative_verify.py` — 15/15 green; existing coordinator/async-pipelined tests still pass). The pre-existing `structured_output/test_engine.py` (fake-import bootstrap, C6) and `test_kv_cache_fp8.py` failures are **unchanged by these fixes** (verified by revert experiment). **C5** (benchmark program) remains open — it requires real multi-node GPU hardware; no numbers were fabricated.

## Status update — C6 + C7 delivered (2026-08-05)
- **C6 (fake-import harness):** a full real-import sweep shows **287/287 core modules and every top-level package now import cleanly WITHOUT `tests/_import_helper` fakes** — the documented "source import bugs" that motivated the fakes are resolved. The only breakage was the legacy `vectorstore/{chroma,pgvector,qdrant}_store.py` importing a nonexistent `VectorStore`; repointed to the real `VectorDBInterface` (audit B21 delete decision still tracked). New real-import suite `tests/real_import/test_real_imports.py` (4 tests incl. the full 287-module sweep) + `tests/scripts/check_real_imports.py` baseline gate, wired as the `real-import-gate` CI job.
- **C7 (coverage frontier):** `tests/scripts/check_core_coverage_frontier.py` computes the 72 zero-test-ref modules (tracked in [[Core Test Coverage Frontier]]), fails CI if the frontier grows, and is wired into the same CI job.
- **Bonus CI fix:** `ci.yml` was **invalid YAML** (pre-existing) — the `benchmark` job's `run: |` block had continuation lines at column 0, so the entire workflow would not load. Extracted that Python into `benchmarks/run_cpu_benchmarks.py`; `ci.yml` now parses and every job (including `real-import-gate`) can run.

## Status update — B1, B2, B3 fixed (2026-08-05)
- **B1** `advanced_scheduling` policy classes are now in-contract with `batch_scheduler` (`set_nodes`, `detect_wan_mode`, constructor kwargs `cost_per_hour_by_node`/`max_power_watts`/…, and `stats()` on all four) — toggling heterogeneous/WAN/cost/energy scheduling and `scheduler.stats()` no longer crash. Contract test `tests/core/test_advanced_scheduling_contract.py` (incl. end-to-end BatchScheduler toggle) — 13/13 green; existing `advanced_scheduling/` and `test_batch_scheduler.py` suites still pass.
- **B2** LoRA merge now applies canonical `alpha/rank` scaling (rank/alpha persisted at `create()`); test `tests/core/test_aether_lora_merge.py`.
- **B3** `TieredMemoryPool` L3 round-trip preserves tensor dtype/shape (magic-prefixed metadata header); tests `tests/core/test_tiered_store_l3_roundtrip.py`.

## Category index (wikilinks)
1. [[Core Audit 01 Strategic & Opportunities]] — Project analysis, moat, competitive + strategic recommendations
2. [[Core Audit 02 Issues & Required Fixes]] — Full catalog: every bug, perf, security, quality, architecture, Tech-debt issue with severity/priority/effort
3. [[Core Audit 03 Enhancements & Modifications]] — Concrete upgrades to existing modules
4. [[Core Audit 04 Advanced Features]] — Differentiating capabilities to build
5. [[Core Audit 05 New Additions]] — New modules / third-party integrations
6. [[Core Audit 06 Verification & Testing]] — Testing + CI strategy
7. [[Core Audit 07 Dead Code & Consolidation]] — Beyond the categories: unwired modules, duplication, consolidation map

## Findings summary

| Category | Count | Critical | High | Medium | Low |
|----------|-------|----------|------|--------|-----|
| bug | 62 | 4 | 18 | 30 | 10 |
| tech_debt | 29 | 0 | 5 | 15 | 9 |
| code_quality | 29 | 0 | 0 | 14 | 15 |
| architecture | 18 | 0 | 6 | 10 | 2 |
| strategic | 17 | 1 | 5 | 11 | 0 |
| test_gap | 19 | 2 | 8 | 7 | 2 |
| security | 9 | 0 | 3 | 5 | 0* |
| performance | 13 | 0 | 1 | 8 | 4 |
| enhancement | 4 | 0 | 1 | 2 | 1 |
| advanced_feature | 1 | 0 | 0 | 1 | 0 |
| integration | 1 | 0 | 0 | 1 | 0 |
| **Total** | **202** | **7** | **47** | **102** | **46** |

*security categories show Mediums where the underlying code is a correctness-of-guarantee or trust-boundary gap with CVSS gradable severity; see [[Core Audit 02 Issues & Required Fixes#Security]].

## Verification ledger
All Critical/High bug+security findings were checked. Independent confirmation:

| Finding | Status |
|---------|--------|
| `coordinator.py:126` — `_subsystem_mgr` read before assignment → `AttributeError` on every `Coordinator()` (P0) | ✅ **CONFIRMED** (this audit, direct read) |
| `kv_cache.py:683` — imports `apply_kv_cache_quantization`/`dequantize_kv_cache` that no longer exist in `quantization_selector.py` → `ImportError` on quantized-KV path | ✅ **CONFIRMED** (this audit, direct read) |
| `carbon_migration.py:206` — `current.gco2_per_kwh` NameError (var is `current_gco2`) → migration never fires | ✅ **CONFIRMED** (this audit, direct read) |
| `structured_output/__init__.py:92` — `_build_token_index` returns `{}` on first call → all-tokens-forbidden mask → immediate EOS | ✅ **CONFIRMED** (this audit, direct read) |
| `atlas_mesh.py:1200` — AtlasMesh/LPSolverRouter key mismatch silently disables multi-objective scoring | ✅ CONFIRMED (adversarial subagent) |
| `compressed_speculative.py:274` — rejection path re-runs identical compressed forward → zero recovery | ✅ CONFIRMED (adversarial subagent) |
| `load_balancer.py:152` — "unhealthy targets never re-probed" | ❌ **REFUTED** — time-based stale-retry is the re-probe; not a defect. See note in Issues catalog. |

Findings not listed above were reported by the read agents with exact-line evidence but were **not** all independently re-read in this session; treat them as high-confidence but prioritize re-verification of any you act on, starting with the release-blockers in [[Core Audit 02 Issues & Required Fixes#Critical]].

## Top 10 actions (priority-ordered)
| # | Action | Category | Severity | Effort |
|---|--------|----------|----------|--------|
| 1 | **Fix `Coordinator.__init__` order** so the platform can start (`self._subsystem_mgr` before line 126) | Bug | Critical/P0 | <1h |
| 2 | **Restore quantized-KV functions** or fix the import in `kv_cache.py:683` | Bug | Critical/P1 | 1–3h |
| 3 | **Make `_build_token_index` synchronous** on first call (structured output) | Bug | Critical/P1 | 1–2h |
| 4 | **Wire (or delete) the verifier** in `async_pipelined_speculative.py` | Bug | Critical/P1 | 3–5h |
| 5 | **Publish a real distributed benchmark program** — the entire GTM rests on unproven 70B claims | Strategic | Critical/P0 | 1–2w |
| 6 | **Separate the fake-import test bootstrap** from a real-import integration suite (114 tests run on `sys.modules` fakes) | Test gap | Critical/P0 | 5–10d |
| 7 | **Fix `dp_inference` DP noise never applied** + unclipped `differential_privacy` — false privacy guarantees | Security | High/P1 | 1–2d |
| 8 | **Wire dormant request-dedup + audit** (currently `None` in coordinator) behind config | Enhancement | High/P1 | 4–8h |
| 9 | **Fix `usage_meter` record-id collision** (silent underbilling) | Bug | High/P1 | 30m |
| 10 | **Ship the Ollama Cluster plugin** as the lead distribution wedge | Strategic | High/P0 | 1–2w |

## Beyond-categories headline
- **>20 core modules are dead/unwired** (never imported by production `src/`) — ~5,500+ LOC of "feature shelf" ([[Core Audit 07 Dead Code & Consolidation]]).
- **At least 7 near-duplicate clusters** (KV-cache ×3, speculative ×5, evaluation monolith vs refactor, grammar ×3, coordinator lifecycle vs `SubsystemManager`, cache-streaming cluster, routing ×3) — the standing production-readiness consolidation decision.
- See [[Core Audit 07 Dead Code & Consolidation]] for the full map.

---
*Generated 2026-08-05. Replaces/extends the earlier `src/distllm/core/DISTLLM_CORE_COMPREHENSIVE_ANALYSIS.md` (2026-07-17) with a complete, per-file read.*
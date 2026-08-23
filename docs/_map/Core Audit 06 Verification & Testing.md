---
tags:
  - core
  - audit
  - testing
date: 2026-08-05
---

# Core Audit 06 — Verification & Testing Strategy

**← [[Core Comprehensive Audit 2026-08-05]]**
Category 6: how to verify the core works. The current suite *looks* strong (6224 test functions / ~69,872 lines, 211 core test files) but has a structural problem: **most of it runs against a fake replica of the package**, not the real `distllm.core`.

> **Status 2026-08-05:** C6 and C7 are **delivered**. A real-import sweep proves **287/287 core modules import cleanly without the fakes**; the new `tests/real_import/` suite (CI: `real-import-gate`) guards the real package graph, and `tests/scripts/check_core_coverage_frontier.py` tracks the **72 zero-coverage modules** ([[Core Test Coverage Frontier]]), failing CI if the frontier grows. The fake bootstrap is kept for pure unit tests; the real-import gate runs separately.

## The core problems (evidence-grounded)

### T1 — The suite fakes the package and proves little  — `tests/_import_helper.py:84` (Critical/P0)
114 of 241 core test files inject **fake `distllm.*` modules into `sys.modules`** and hand-write stub replacements for real classes (`JSONSchemaConstraint` → FSM stub, `validate_structured_output` → identity). The scheduler / advanced_scheduling / structured_output / vectorstore suites — the green ones — exercise **none** of the real `__init__.py` exports or cross-module wiring. This is exactly why the release-blocker import bugs in [[Core Audit 02 Issues & Required Fixes#Critical|C1–C3]] shipped green. Several real import bugs (models.partitioner vs partition_planner, core→dist circular chain) are "worked around" in the harness instead of fixed.

### T2 — 78 of 287 core modules have zero test references  — `coordinator_subsystem.py`, `evaluation/runner.py`, `dp_inference/accounting.py`, … (Critical/P0)
27% of core ships with no direct or aggregate coverage, including the **production coordinator startup path**, the **eval-quality gate**, **DP crypto math**, the **KV memory-management layer**, and **chaos/failover state machines**. Highest-risk holes by subsystem: coordinator wiring (`coordinator_config_wiring`, `_subsystem`, `_request`, `_election`, `cli`); the whole `evaluation/*` pipeline; `dp_inference.{accounting,config,mechanisms}`; `kv_cache_{manager,paged,migration,replication,serialization,metrics,adaptive_quantizer}`; `kraken_chaos`/`spot_*`/`aegis_compliance`/`node_recovery`; `vllm_node`/`wisp_wasm`/`faas_7b`/`cortex_multimodal`/`hydra_diffusion`/`voyager_multimodal`; `priority_heap`/`sentinel_qos`/`starvation_monitor`/`pareto_optimizer`/`block_eviction_policy`; `stats_collector`/`monitor`/`latency_tracker`/`activation_profiler`/`aria_autoscaler`/`autoq`/`pulse_performance_model`/`predictive_failure`.

### T3 — Trivial smoke tests inflate the counts  — `tests/core/vectorstore/test_vectorstore.py`
Files like `test_vectorstore.py` assert only that `__init__` stored a constructor arg (`assert store._dims == 384`) — no upsert/query/eviction behavior. Never actually storing/retrieving a vector while CI reports green gives users false confidence. Same pattern in `test_pipeline_executor.py` (6 tests/8 asserts), `test_shadow_eval.py`, `test_usage_meter_quota.py`, `test_quant_dequant.py`, `test_kv_cache_disk.py`.

### T4 — No reproducible performance baseline  — `tests/core/bench_cache_lookup_latency.py`
The only core perf test hard-codes **machine-dependent absolute wall-clock thresholds** (P50<1.0ms, P99<10ms, ops>100k) with no stored baseline → passes on fast machines, fails on CI runners. `performance_baseline.py`/`cache_bench.py`/`latency_tracker.py` are zero-coverage, and no `@pytest.mark.benchmark` marker exists. The project has no reproducible throughput/latency/cache-hit/acceptance-rate baseline at all.

### T5 — pytest markers declared but never used  — `pytest.ini`
`integration/e2e/slow/memory/chaos/benchmark/security/property/sdk/cli` markers exist but only **one** test in `tests/core` uses a marker. There is no fast gate that deselects torch/GPU/network tests; every workflow runs overlapping unmarked suites.

### T6 — Environmental vs real failures are conflated  — `test_node_recovery.py`
61 core test files import `torch`, 81 use time/sleep, 114 depend on the order-sensitive fake bootstrap, with **no GPU skipif, no clock injection, no retry** — so the green/red set differs between laptop / CI / GPU box, and a real logic regression is indistinguishable from an environmental failure in the same job.

## Recommended strategy (5 layers, mapped to CI tiers)

### Layer 1 — Unit (per-module, deterministic, CPU-only)
Cover the zero-frontier modules that are pure/stateless and trivially testable first: `priority_heap` invariants, `dp_inference.accounting` epsilon/δ with hand-computed vectors, `pareto_optimizer` dominance, `block_eviction_policy` ordering, `latency_tracker` percentiles, `stats_collector`, `starvation_monitor`. Inject clocks (`time` via a `Clock` seam), gate GPU with `skipif(ucuda())`.

### Layer 2 — Integration (real-import, CPU-only wiring) — the missing layer
A new suite that imports the **real package** and runs fix-staged smoke paths: `from distllm.core.coordinator import Coordinator` construction, `structured_output.JSONSchemaConstraint` first-call mask, quantized-KV `append/get_all`, scheduler→policy DI, request→audit/dedup. Fix the ~4 documented source import bugs first; **then** delete the stub-replacement code in `_import_helper.py`. Add a CI job asserting `python -c 'from distllm.core.coordinator import Coordinator'` succeeds.

### Layer 3 — Benchmark (reproducible baselines)
Replace absolute thresholds with a **baseline artifact** (JSON with hardware fingerprint + seeds + N≥30), and add `@pytest.mark.benchmark` markers: throughput (tok/s), TTFT/ITL P50/P99, cache hit-rate, speculative acceptance-rate per strategy. Wire into a `benchmark-regression` gate that **fails CI on regression** (build the baseline for real distributed numbers first — see [[Core Audit 05 New Additions#N1]]).

### Layer 4 — Security (+ fuzz)
Batch: `bandit` + `safety` + `pip-audit` on core; targeted fuzz on `constrained_decoder`/`grammar_decoder` masks and `structured_output` schemas; tests for `secret_manager` lifecycle (rotate/revoke/leak-on-error), `plugin_sandbox` capability enforcement, and the DP no-guarantee guards ([[Core Audit 02 Issues & Required Fixes#S1]]). Extend the existing `tests/security/` API coverage to reach these core primitives.

### Layer 5 — Chaos + UAT
Assert the **core** kill/failover/eviction **state machines in-process** (CPU-only, deterministic) — `spot_failover`, `node_recovery` recovery/replay, `ha_coordinator` peer re-admission, `semantic_cache` invalidation — rather than only via infra-hungry `tests/chaos/`. UAT: a scriptable `distllm cluster demo` 2-node run with time-to-first-token metric (see [[Core Audit 05 New Additions#N3]]).

## CI pipeline (4 tiers, mapped to core)
| Tier | Trigger | Contents | Duration |
|------|---------|----------|----------|
| **Fast / PR** | Every PR | lint (ruff, mypy), import-wiring smoke (`from distllm.core.coordinator import Coordinator`, `from distllm.core.structured_output import JSONSchemaConstraint`), CPU unit tier `-m 'not integration'` | <5 min |
| **Merged** | On merge | full unit + integration (real-import wiring), security (bandit/safety), coverage-frontier gate (fail if a core module has no test ref) | <15 min |
| **Deep / nightly** | Nightly | benchmark suite with baselines (acceptance-rate, cache hit-rate, latency), fuzz short, DP/secret tests | <40 min |
| **Soak / weekly** | Weekly | load (locust), soak (long-run), chaos (kill/failover), GPU tier | <60 min |

## Immediate backfill priority (the 78-module frontier)
1. `coordinator_config_wiring` + `coordinator_subsystem` (startup path, and the recently-changed `_start_subsystem`)
2. `evaluation/runner` + `scorers` (the model-quality gate)
3. `dp_inference.mechanisms` + `accounting` (champion math, hand-computed vectors)
4. `kv_cache_manager` + `kv_cache_paged` + `kv_cache_serialization` (memory-management hot path, round-trips)
5. `spot_failover` + `node_recovery` + `ha_coordinator` (failover state machines, in-process)

---
**← [[Core Comprehensive Audit 2026-08-05]]** · Next: [[Core Audit 07 Dead Code & Consolidation]]
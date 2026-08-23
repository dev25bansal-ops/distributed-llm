---
tags:
  - core
  - audit
  - architecture
  - techdebt
date: 2026-08-05
---

# Core Audit 07 — Dead Code, Unwired Modules & Consolidation

**← [[Core Comprehensive Audit 2026-08-05]]**
Category 7 (beyond the six requested): the systemic finding of the whole audit. A large fraction of `src/distllm/core/**` — several thousand LOC — **is never imported by production code**, and much of what *is* imported is a near-duplicate copy of a sibling. This is the standing production-readiness consolidation decision.

## Headline numbers
- **20+ modules are dead/unwired** (no production `src/` importer) — ~5,500+ LOC.
- **At least 7 near-duplicate clusters** (KV cache ×3, speculative ×5, evaluation mono-vs-refactor, grammar ×3, coordinator lifecycle vs `SubsystemManager`, cache-streaming ×4, routing ×3).
- **Production-critical collaborators hardcoded to `None`** in `Coordinator.__init__` (request dedup, audit, prompt-cache, preemption policy → ~1,000 LOC of tested code never runs).

## 1. Near-duplicate clusters → pick one canonical home, delete the rest

### KV cache — `kv_cache.py:781` (High)
`KVCacheManager` is implemented **three times** (`core/kv_cache.py`, dead `core/kv_cache_manager.py`, real `core/scheduler/kv_cache_manager.py`). `kv_cache_manager.py` says "Extracted from kv_cache.py to reduce that file below the 800-line ceiling" — but `kv_cache.py` is still 1,254 LOC and the extracted siblings (`kv_cache_adaptive_quantizer`, `kv_cache_paged`, `kv_cache_serialization`, `kv_cache_migration`) are imported by **nothing**. Byte-identical bodies → future fixes silently diverge.
**Decision:** The load-bearing `KVCacheManager` is `core/scheduler/kv_cache_manager.py`; `kv_cache.py`'s is test-only. Split `kv_cache.py` for real, delete the dead siblings (they duplicate originals already inside `kv_cache.py`), add a CI import-test asserting the duplicate names are gone.

### Speculative decoding — 5 variants share logic (High)
`speculative_decoder`, `tree_speculative_decoder`, `distributed_speculative`, `compressed_speculative`, `async_pipelined_speculative` (+ `SelfSpeculativeDecoder`/`TreeDraftSpeculativeDecoder` classes referenced only in docstrings, + `multi_draft_verifier`, `draft_tree`, `draft_model_router`, `draft_quality_scorer`, `mtp_head`). They share the `prefix_len` off-by-one class of bug ([[Core Audit 02 Issues & Required Fixes#B19]]).
**Decision:** converge on a `SpecDecoderBase` mixin (already started), one per strategy; a parity test matrix (acceptance-rate, latency) across strategies.

### Evaluation — monolith vs refactor (High)
Production imports `evaluation_harness.py` (1530 LOC). The extracted `evaluation/*` refactor (runner/scorers/worker/db/…) is **never imported** and carries a `_SecretStr`-wrapper API-key bug ([[Core Audit 02 Issues & Required Fixes|F67]]).
**Decision:** make `evaluation/` the single source of truth, fix the key bug, re-point `api/routes/eval.py` + `api/services/eval_service.py`, delete the monolith.

### Grammar-constrained decoding — 3 modules + never-invoked hook (High)
`grammar_constrained.py` (outlines-backed), `grammar_constrained_draft.py` (claims wiring that doesn't exist — `speculative_decoder.py` has zero `grammar` refs), `grammar_decoder.py` (naive GBNFFSM, permits EOS in any state). `_patch_schema_constrained_decoder` is defined but never called.
**Decision:** keep `grammar_constrained` as the single path, invoke the patch from `constrained_decoder`, wire draft-mask with a real test, retire/fix `grammar_decoder`.

### Cache-streaming — 4 implementations, 1 enforced contract (Medium)
`prefix_cache` (dist, wired), `radix_tree_cache`, `redis_prompt_cache`, `prompt_caching_service` diverge; `protocols.ICacheBackend` declared but unused.
**Decision:** adopt `ICacheBackend`; implement for prefix/radix/Redis; one backend from config; delete the rest (see [[Core Audit 03 Enhancements & Modifications#E4]]).

### Routing — 3 overlapping routers (Medium)
`unified_router`, `unified_sla_router`, `CrossCloudRouter` overlap; `CostComparison`/`CostOptimizer` consume the same data independently.
**Decision:** give `unified_router` the carbon-aware engine and delete the older scatter ([[Core Audit 01 Strategic & Opportunities|F52]]).

### Coordinator lifecycle — `SubsystemManager` (620 LOC) is dead (High)
`coordinator_subismsystem.py`'s `SubsystemManager` is stored at `coordinator.py:217` and **never called**; `Coordinator.start()/stop()` re-implement the identical lifecycle inline. Divergent copies of shutdown logic.
**Decision:** delegate `start()/stop()` to the manager and delete the inline copies, or delete the manager. (This is also where `spot_failover`/predictive-failure wiring should land.)

## 2. Dead / unwired modules — wire, feature-flag, or delete

### Cluster A — "feature shelf" (High) — `pipeline_executor.py:110`
`pipeline_executor` (PlanCompiler/PipelineExecutor/ExecutionPlan), `pipeline_composer`, `pareto_optimizer`, `performance_alerts`, `performance_baseline`, `persistent_store` (~486 LOC), `plugin_marketplace`, `predictive_cache_warming`, `predictive_failure`, plus deprecated shims `pipeline_orchestrator` and `predictive_cache`. Exist only as self-contained modules exercised by tests.

### Cluster B — (High) — `starvation_monitor.py:16`
`starvation_monitor`, `stats_collector`, `step_processor`, `synapse_debugger`, `synth_data_generator`, `state_replication` (StateReplicationStore/TopologyStateStore), `spot_failover`, `spot_forecasting`, `straggler_alerts`, `straggler_aware_scheduler`. Several are literally re-inlined as private `_check_starvation`/`_aging_boost` in `batch_scheduler.py` — the "extraction" copied logic but left the originals unused.

### Cluster C — stubs claimed as features
`cortex_multimodel.py` (939 LOC, dead, serves empty `dummy_kv`), `kv_cache_marketplace.py`, `multimodal_engine.py` (drops media tensors), `voyager_multimodal.py`, `wisp_wasm.py` (WASM capability model unimplemented), `hydra_diffusion.py`, `faas_7b.py`, `vllm_node.py`, `webgpu_manager.py`.

### Cluster D — legacy/unreachable
`vectorstore/{chroma,qdrant,pgvector}_store.py` (import nonexistent `VectorStore`, ImportError — [[Core Audit 02 Issues & Required Fixes#B21]]).

## 3. Production wiring gaps (tested code hardcoded off)
- `Coordinator.__init__` sets `_request_fingerprinter=None`, `_request_auditor=None`, `_preemption_policy=None`, `_prompt_cache_service=None` — dedup, coalescing, audit, and prompt-cache never run ([[Core Audit 02 Issues & Required Fixes#B14]]).
- Broken DI injection: `advanced_scheduling` policy classes out-of-contract with `batch_scheduler` (WAN/energy/cost/heterogeneous crash on toggle) — [[Core Audit 02 Issues & Required Fixes#B1]].
- `core/__init__.py:70` `_register()` grows to 242 modules with no ownership; the "wiring contract" that would catch dead exports is absent.

## 4. Decision framework (wire vs feature-flag vs delete)
| Triage | Modules | Rationale |
|--------|---------|----------|
| **Wire** (high differentiation, tested) | predictive-failure → `health_manager`; pareto → `model_router`; spot-failover/cost-forecast → coordinator subsystems; sentinel/qos + route-audit; request dedup + audit; prefix-cache via `ICacheBackend` | [[Core Audit 03 Enhancements & Modifications]] E1/E2/E7/E8; real moat |
| **Feature-flag / EXPERIMENTAL** | marketplace, DP, federated, watermarking, A/B, cortex, multimodel, wisp/wasm, hydra | Shrink the 242-module surface ([[Core Audit 01 Strategic & Opportunities#F180]]) |
| **Delete** | dead siblings that duplicate originals (kv_cache_manager etc.), legacy `*_store.py`, `synapse_debugger`, over-claimed stubs | Zero current users, duplicated logic |
| **Block** | new core LOC without a production importer + integration test | Stop the "shiny but inert" accumulation |

## 5. Guardrail to stop the rot
Add a **CI import-graph guard** that fails when a `core/` module has no production `src/` importer (or isn't feature-flagged experimental). This single check would have caught clusters A/B/C before they accumulated.

---
**← [[Core Comprehensive Audit 2026-08-05]]** · Back to top: [[Core Audit 02 Issues & Required Fixes]]
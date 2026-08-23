---
tags:
  - core
  - audit
  - testing
date: 2026-08-05
---

# Core Test Coverage Frontier (tracked)

**← [[Core Comprehensive Audit 2026-08-05]]** · Part of [[Core Audit 06 Verification & Testing]] (finding C7)

Live list of `src/distllm/core/**` modules with **zero direct test references** (no `import`/`from`/`load_module`/`spec_from_file_location` in any `tests/` file). Recompute with:

```bash
PYTHONPATH=src python tests/scripts/check_core_coverage_frontier.py   # read-only
PYTHONPATH=src python tests/scripts/check_core_coverage_frontier.py --write  # after legitimately covering modules
```

The CI gate `real-import-gate` runs the read-only checker and **fails if this frontier grows**. Backfill work shrinks it and re-records the baseline.

**Current frontier: 72 modules (2026-08-05 baseline).**

## [core] — 59
`ab_test_coordinator, activation_profiler, adaptive_cache_compressor, aegis_compliance, aether_federated, aria_autoscaler, atlas_mesh, auto_discovery, autoq, bargaining_engine, batch_builder, block_affinity_tracker, block_eviction_policy, cache_bench, cache_template_warmer, coordinator_cli, coordinator_config_wiring, coordinator_request, cortex_multimodel, debug, dynamic_memory_budget, faas_7b, gaia_cache, hydra_diffusion, kraken_chaos, kv_cache_adaptive_quantizer, kv_cache_manager, kv_cache_metrics, kv_cache_migration, kv_cache_paged, kv_cache_serialization, monitor, multi_draft_verifier, neural_partition_optimizer, node_recovery, notification_manager, pareto_optimizer, performance_alerts, pipeline_orchestrator, preference_learning, prefix_cache, priority_heap, pulse_performance_model, rebalancer, route_audit, sentinel_qos, spot_failover, spot_forecasting, starvation_monitor, stats_collector, step_processor, straggler_alerts, straggler_detector, synapse_debugger, synth_data_generator, tree_speculative_decoder, vllm_node, voyager_multimodal, wisp_wasm`

## [dp_inference] — 3
`accounting, config, mechanisms`

## [evaluation] — 9
`constants, db, formatters, loaders, models, report, runner, scorers, worker`

## [vectorstore] — 1
`factory`

## Notes
- Several entries are **deprecated thin re-exports to `dist`** (`prefix_cache`, `rebalancer`, `straggler_detector`, `vllm_backend`) — low priority; the `dist` targets have tests. Prefer to test the canonical `dist` modules and delete/point the shims.
- Modules that are **dead/unwired** (see [[Core Audit 07 Dead Code & Consolidation]]) should be decided (wire vs delete) rather than blindly given tests.

## Backfill priority
1. **Production startup/wiring** — `coordinator_config_wiring`, `coordinator_request`, `coordinator_cli` (the C1-class bug lives here; these are the highest-risk holes).
2. **Model-quality gate** — `evaluation/runner`, `evaluation/scorers`, `evaluation/db` (PPL/quality scoring that decides what gets served).
3. **DP math** — `dp_inference/mechanisms`, `dp_inference/accounting` (champion Laplace/Gaussian noise + ε/δ with hand-computed vectors).
4. **KV memory-management hot path** — `kv_cache_manager`, `kv_cache_paged`, `kv_cache_serialization` (block alloc/free, paging, round-trips).
5. **Failover state machines (in-process)** — `spot_failover`, `node_recovery`, `kraken_chaos`, `aegis_compliance` (deterministic CPU-only, not infra-hungry `tests/chaos/`).
6. **Data structures** — `priority_heap` (push/pop/promote invariants), `block_eviction_policy` (LFU/LRU ordering), `starvation_monitor`.

---
← Back to [[Core Audit 06 Verification & Testing]]
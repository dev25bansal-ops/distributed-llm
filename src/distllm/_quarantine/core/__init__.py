"""Quarantine zone for unwired modules (2026-09-01 dead-code census).

Modules here had **zero imports** anywhere in the repository (verified by
both module-name and class-name grep across ``src/``, ``tests/``,
``integrations/``, ``sdk/``, ``examples/``, ``scripts/``, ``benchmarks/``).
They are preserved for possible future wiring, but excluded from the
runtime package surface.

DO NOT import from here in production code. If a module becomes needed:
1. Move it back to its original package.
2. Wire it into a real call site.
3. Add a test that exercises it.
4. Remove this entry from the quarantine README.

**Expiry policy (O-10):** this directory is scheduled for deletion on
**2026-10-01** (30-day review date). Before that date, either wire a
module back in (steps above) or delete it — unpromoted modules are
removed wholesale, not migrated again. The 2026-10-01 sweep is itself a
release-blocker item in
``docs/_audit/exhaustive-2026-08-31/ISSUES_CATALOG_CURRENT.md`` (§4, O-10).

| Module | Original LOC | Notes |
|---|---|---|
| block_eviction_policy | 363 | Eviction policy variants never wired into cache tiers |
| block_affinity_tracker | — | PagedAttention affinity tracking, unreferenced |
| kv_cache_serialization | 182 | Legacy serialization, superseded by safetensors path |
| kv_cache_migration | 137 | One-shot migration shims for old cache formats |
| kv_cache_adaptive_quantizer | 168 | Adaptive quant thresholds, never wired into kv_cache |
| kv_cache_metrics | 205 | Prometheus gauges never registered |
| model_hub_enhanced | — | Enhanced model-hub wrapper; zero production refs (2026-09-12 ARCH-1) |
| neural_partition_optimizer | 1199 | Neural cost-model partitioner; complete but unwired (2026-09-12 ARCH-1) |
| dynamic_memory_budget | 179 | Memory budget planner, unreferenced |
| cache_template_warmer | — | Template pre-warming, unreferenced |
| auto_discovery | 351 | mDNS discovery, superseded by cli/devices.py |
| activation_profiler | 172 | Activation-size profiling, unreferenced |

Deleted outright (zero value, confirmed dead): ``wisp_wasm``, ``spot_forecasting``,
``spot_failover``, ``synapse_debugger``, ``faas_7b``, ``performance_alerts``,
``vllm_node``, ``straggler_alerts``, ``starvation_monitor``, ``route_audit``,
``cache_bench`` — see ``docs/_audit/exhaustive-2026-08-31/`` for the census.
"""

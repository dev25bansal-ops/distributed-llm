# CLAUDE.md — Core Engine

## Your Scope
You have ownership of `src/distllm/core/` — the inference engine, scheduler, KV cache, coordinator, speculative decoding, cost tracking, and everything in `core/`.

## Do NOT Touch
- `src/distllm/dist/` — distributed layer (handled by another instance)
- `src/distllm/api/` — API server (handled by another instance)
- Any other module outside `core/`

## Key Files

| File | Purpose |
|------|---------|
| `coordinator.py` | Main orchestrator — generation, HA, state replication |
| `inference_engine.py` | Text generation — local, distributed, speculative strategies |
| `batch_scheduler.py` + `scheduler/` | Continuous batching, iteration budgets |
| `kv_cache.py` | KV cache management, FP8 quantization |
| `cache_manager.py` + `cache_*.py` | Multi-tier caching, prefix cache, radix tree |
| `model_router.py` + `load_balancer.py` | Smart routing and load distribution |
| `speculative_decoder.py` | Self-speculation (Medusa/EAGLE) |
| `distributed_speculative.py` | Remote draft + multi-draft speculative decoding |
| `structured_output/` | JSON mode, schema validation |
| `advanced_scheduling/` | Heterogeneous, WAN, energy-aware scheduling |
| `cost_tracker.py` + `streaming_cost.py` | Usage metering and cost tracking |
| `auto_discovery.py` | mDNS-based node discovery (NEW) |
| `model_sizing.py` | Model size estimation (NEW) |
| `coordinator_config.py` | Configuration schema |
| `health_manager.py` | Health probes, failover |
| `cluster_manager.py` | Node registration, topology |
| `memory_defragmenter.py` | PagedAttention block compaction |
| `resource_manager.py` | gRPC connection management |

## Current State
- All security fixes applied
- All performance optimizations applied
- Micro-batch 1F1B scheduling with backpressure
- Continuous HA state replication
- FP8 per-step quantization in KV cache

## Commands
- `python -m pytest tests/core/ -v` — run core tests
- `python -m pytest tests/core/test_kv_cache_fp8.py -v` — KV cache tests
- `python -m pytest tests/core/test_coordinator_state_replication.py -v` — HA replication tests
- `python -m pytest tests/core/test_node_recovery.py -v` — recovery tests

---
tags:
  - core
  - engine
---
# Core Engine

**Location:** `src/distllm/core/` — **5.4 MB, ~55K LOC, ~100 files**

**Commands:** `python -m pytest tests/core/ -v`

## Key Files
| File | LOC | Purpose |
|------|-----|---------|
| `coordinator.py` | 1474 | Main orchestrator — generation, HA, state replication |
| `inference_engine.py` | 889 | Text generation — local, distributed, speculative strategies |
| `batch_scheduler.py` | 1333 | Continuous batching, iteration budgets |
| `kv_cache.py` | 1148 | KV cache management, FP8 quantization |
| `distributed_speculative.py` | 1358 | Remote draft + multi-draft speculative decoding |
| `model_router.py` | 931 | Smart model routing |
| `cost_tracker.py` | 867 | Usage metering and cost tracking |
| `structured_output/` | ~300 | JSON mode, schema validation |

## Subsystems
- **Scheduler** — `batch_scheduler.py` + `scheduler/` — layered budget system (Heterogeneous → WAN → Energy → Sarathi → Base)
- **KV Cache** — `kv_cache.py`, `cache_manager.py`, `prefix_cache.py`, `radix_tree_cache.py` — FP8 per-step quantization, defragmentation
- **Speculative Decoding** — `speculative_decoder.py` (self-speculation via Medusa/EAGLE heads), `distributed_speculative.py` (remote draft + multi-draft)
- **Cost Tracking** — `cost_tracker.py`, `streaming_cost.py`, `cost_middleware.py` — per-tenant, per-request cost tracking
- **Scheduling** — `advanced_scheduling/` — heterogeneous, WAN, energy-aware, predictive, cost-aware, federated

## Dependencies → [[docs/_map/02 Distributed Layer]], [[docs/_map/03 API Server]]

## Recent Work
- ✅ 1F1B micro-batched scheduling with backpressure
- ✅ Continuous HA state replication with 1s push interval
- ✅ Self-speculation (Medusa/EAGLE-style draft heads)
- ✅ FP8 per-step KV cache quantization
- ✅ Automatic recovery replay from checkpoints
- ✅ Per-request privacy projections
- ✅ Constrained decoder for structured output

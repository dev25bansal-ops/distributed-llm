---
tags:
  - dist
  - distributed
---
# Distributed Layer

**Location:** `src/distllm/dist/` — **4.0 MB, ~60 files**

**Commands:** `python -m pytest tests/dist/ -v`

## Subsystems
| Subsystem | Files | Purpose |
|-----------|-------|---------|
| `pipeline/` | 8 | Orchestrator, transport, serialization, simulator |
| `partition/` | 20+ | Partitioner, optimizer, cost model, quantization |
| `p2p/` | 4 | Discovery, gossip, router, transport |
| `backends/` | 3 | llama.cpp, vLLM, Ray integrations |
| `scheduling/` | 3 | Batcher, classifier, iteration scheduler |

## Key Files
| File | LOC | Purpose |
|------|-----|---------|
| `pipeline/orchestrator.py` | 541 | 1F1B pipeline scheduling + dynamic micro-batch sizing |
| `worker.py` | 615 | gRPC worker node |
| `recovery.py` | 658 | Node failure recovery + checkpoint replay |
| `straggler.py` | 737 | Straggler detection (EMA, Holt-Winters, MAD) |
| `federation.py` | 842 | Cross-cluster federation |
| `fsdp.py` | ~295 | FSDP weight sharding (NEW) |
| `privacy.py` | ~150 | Per-request privacy projections (NEW) |
| `parallel.py` | 1050 | Parallel execution management |
| `attention.py` | 1636 | Attention kernel implementations |
| `cross_cluster.py` | 199 | KV cache protobuf transfer |

## 1F1B Scheduling
```
bubble_ratio = (num_stages - 1) / (num_micro_batches + num_stages - 1)
Without micro-batching: 75% for 4 stages
With 16 micro-batches: 16%
```

## Dependencies → [[docs/_map/01 Core Engine]]

## Recent Work
- ✅ 1F1B scheduling with backpressure (`asyncio.Semaphore`)
- ✅ Dynamic micro-batch sizing from straggler feedback
- ✅ Memory-bandwidth-bound throughput model (vs FLOP roofline)
- ✅ FSDP-style weight sharding across nodes
- ✅ Auto mixed-precision pipeline (per-layer FP8/INT8/FP16)
- ✅ Multi-draft speculative decoding (`_query_all_drafts`, `draft_voting`)
- ✅ KV cache binary protobuf transfer (10-50x faster)

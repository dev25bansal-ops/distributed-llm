# CLAUDE.md — Distributed Layer

## Your Scope
You have ownership of `src/distllm/dist/` — pipeline orchestration, model partitioning, P2P networking, federation, straggler detection, recovery, gRPC services, and everything in `dist/`.

## Do NOT Touch
- `src/distllm/core/` — core engine (handled by another instance)
- `src/distllm/api/` — API server (handled by another instance)
- Any other module outside `dist/`

## Key Files

| File | Purpose |
|------|---------|
| `pipeline/orchestrator.py` | Pipeline parallelism + 1F1B micro-batched scheduling |
| `pipeline/transport.py` | Tensor transport |
| `pipeline/serialization.py` | Protobuf tensor serialization (FP8, CUDA streams) |
| `pipeline/simulator.py` | Pipeline simulation |
| `partition/partitioner.py` | Model partitioning across nodes |
| `partition/optimizer.py` | DP partition solver |
| `partition/quantization_tuner.py` | Auto mixed-precision pipeline |
| `partition/cost_model.py` | Partition cost modeling |
| `p2p/gossip.py` | Gossip protocol |
| `p2p/discovery.py` | Peer discovery |
| `federation.py` | Cross-cluster federation |
| `wide_area.py` | WAN pipeline |
| `cross_cluster.py` | KV cache protobuf transfer |
| `recovery.py` | Node failure recovery + checkpoint replay |
| `straggler.py` | Straggler detection |
| `worker.py` + `node_client.py` + `node_service.py` | gRPC worker nodes |
| `fsdp.py` | FSDP weight sharding (NEW) |
| `privacy.py` | Per-request privacy projections |
| `rebalancer.py` | Layer rebalancing |

## Current State
- 1F1B scheduler with dynamic micro-batch sizing
- Memory-bandwidth-bound throughput estimation
- FSDP weight sharding across nodes
- Auto mixed-precision pipeline per-layer
- Straggler-aware dynamic batch sizing
- Multi-draft speculative decoding

## Commands
- `python -m pytest tests/dist/ -v` — run distributed layer tests
- `python -m pytest tests/dist/pipeline/test_1f1b_scheduling.py -v` — 1F1B tests
- `python -m pytest tests/dist/test_federation_heartbeat.py -v` — federation tests

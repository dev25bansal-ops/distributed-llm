# ADR-0001: Pipeline Parallelism as Primary Distribution Strategy

**Date:** 2024-01-15
**Status:** Accepted
**Deciders:** Core team

## Context

We need to distribute large language models across multiple consumer GPUs. The main approaches are:

1. **Tensor Parallelism (TP)**: Split each layer across GPUs (requires high-bandwidth interconnect)
2. **Pipeline Parallelism (PP)**: Split layers sequentially across GPUs (works over any network)
3. **Expert Parallelism (EP)**: For MoE models, route tokens to different experts

Consumer GPUs are connected via:
- NVLink (same machine): 600+ GB/s
- PCIe: 32 GB/s
- Home LAN: 1-10 Gbps
- WAN (internet): 100 Mbps - 1 Gbps

## Decision

We chose **Pipeline Parallelism** as the primary distribution strategy because:

1. **Works over any network**: TP requires >100 GB/s bandwidth; PP works even over 1 Gbps LAN
2. **Consumer-friendly**: Most users have GPUs on different machines, not NVLink-connected
3. **Predictable latency**: Each node processes a fixed set of layers
4. **Fault-tolerant**: If one node fails, we can redistribute its layers

We also support TP within a single node (multiple GPUs with NVLink) and EP for MoE models.

## Consequences

### Positive
- Works on consumer hardware (RTX 3060, 4090, etc.)
- Supports WAN inference (internet-scale distributed inference)
- Enables "friends pooling GPUs" use case
- Predictable performance characteristics

### Negative
- Higher latency than TP for single-machine multi-GPU setups
- Each node must hold a complete layer (no intra-layer parallelism)
- Pipeline bubble overhead for small batch sizes

### Mitigations
- Support TP within nodes (NVLink-connected GPUs)
- Implement overlap scheduling to hide communication latency
- Use speculative decoding to amortize pipeline latency

## Related ADRs
- ADR-0002: WAN Optimization Strategy
- ADR-0003: Speculative Decoding Architecture

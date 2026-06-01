# ADR-0003: Distributed Speculative Decoding Architecture

**Date:** 2024-03-10
**Status:** Accepted
**Deciders:** Core team

## Context

Autoregressive generation is slow because each token requires a full forward pass. Speculative decoding uses a fast draft model to propose candidates, then verifies them with the target model in a single pass.

Key question: How to do speculative decoding across distributed nodes?

## Decision

We implemented a **Distributed Draft-as-a-Service (DaaS)** architecture:

1. **Draft Model Routing**: Draft models run on separate nodes (CPU or edge devices)
   - Reduces GPU memory pressure on target model nodes
   - Enables heterogeneous hardware (draft on CPU, target on GPU)

2. **Fleet Mode**: Multiple draft models with different capabilities
   - Auto-selects best draft model based on acceptance rate
   - Falls back to local drafting when remote unavailable

3. **Adaptive Candidates**: Dynamically adjust number of draft tokens
   - High acceptance rate → more candidates (speculate further)
   - Low acceptance rate → fewer candidates (reduce waste)

4. **Multi-Draft Verification**: Multiple draft models propose simultaneously
   - Target model verifies all candidates in single pass
   - Accepts first matching token from any draft

## Consequences

### Positive
- 2-4x throughput improvement for interactive workloads
- Draft models can run on cheap hardware (CPU, old GPUs)
- Fleet mode provides redundancy and load balancing
- Adaptive candidates optimize for varying workloads

### Negative
- More complex orchestration (draft model lifecycle)
- Additional network hops for remote draft models
- Memory overhead for draft model loading

### Mitigations
- Draft models are small (1-3B parameters)
- Local fallback when remote drafts unavailable
- LRU eviction for draft model memory management

## Related ADRs
- ADR-0001: Pipeline Parallelism
- ADR-0002: WAN Optimization Strategy

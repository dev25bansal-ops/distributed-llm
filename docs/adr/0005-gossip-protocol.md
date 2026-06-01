# ADR-0005: Gossip Protocol for KV Cache Discovery

**Date:** 2024-05-15
**Status:** Accepted
**Deciders:** Core team

## Context

When multiple nodes have cached KV states for different prompts, we need to discover which nodes have which cache entries to route requests efficiently.

Centralized cache registry would be a single point of failure. We need a decentralized approach.

## Decision

We implemented a **CRDT-based gossip protocol**:

1. **Anti-Entropy Gossip**: Nodes exchange cache advertisements periodically
   - Each round contacts 3 random peers (fanout=3)
   - Sends only delta changes since last exchange
   - Convergence time: O(log(N)) rounds

2. **CRDT Semantics**: Conflict-free replicated data types
   - G-Set (grow-only set) for cache entries
   - LWW-Register (last-writer-wins) for metadata
   - Vector clocks for causal ordering
   - Tombstones for deletions

3. **Bloom Filter Pre-check**: Skip exchange when no changes
   - Reduces unnecessary network traffic
   - Quick change detection without full comparison

## Consequences

### Positive
- Decentralized (no single point of failure)
- Self-healing (automatically recovers from partitions)
- Eventually consistent (all nodes converge)
- Low overhead (delta-only propagation)

### Negative
- Eventual consistency (not immediate)
- Stale data possible during convergence
- Memory overhead for vector clocks and tombstones

### Mitigations
- Configurable convergence interval
- Tombstone TTL for automatic cleanup
- Fallback to direct query when gossip data stale

## Related ADRs
- ADR-0001: Pipeline Parallelism
- ADR-0002: WAN Optimization Strategy

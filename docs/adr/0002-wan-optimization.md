# ADR-0002: WAN Optimization Strategy

**Date:** 2024-02-20
**Status:** Accepted
**Deciders:** Core team

## Context

Pipeline parallelism over WAN links (internet) introduces high latency:
- Typical RTT: 50-200ms per hop
- 4-node pipeline: 200-800ms per token
- This is too slow for interactive chat (need <200ms/token)

## Decision

We implemented multiple WAN optimizations:

1. **Token Accumulation**: Buffer N decode steps before sending across WAN
   - Amortizes RTT over N tokens instead of 1
   - N is adaptive based on measured RTT

2. **QUIC Transport**: UDP-based protocol instead of TCP/gRPC
   - 0-RTT connection establishment
   - No head-of-line blocking
   - Better packet-loss recovery

3. **Speculative WAN Decoding**: Draft model generates candidates locally, verifies remotely
   - Reduces WAN round-trips by 3-5x
   - Only sends verification request, not each token

4. **Adaptive Batching**: Adjust accumulation window based on measured RTT
   - Higher RTT → larger accumulation window
   - Auto-calibrates from recent measurements

## Consequences

### Positive
- WAN inference becomes usable (5-15 tokens/sec over internet)
- QUIC provides better loss recovery than TCP
- Adaptive batching optimizes for varying network conditions

### Negative
- Increased memory usage for token accumulation buffer
- More complex error handling for async operations
- QUIC requires additional dependency (aioquic)

### Mitigations
- Memory-bounded accumulation buffers
- Graceful degradation to gRPC when QUIC unavailable
- Configurable limits on accumulation window

## Related ADRs
- ADR-0001: Pipeline Parallelism
- ADR-0003: Speculative Decoding Architecture

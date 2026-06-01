# Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                        │
│  /v1/chat/completions  /v1/completions  /v1/embeddings  /ws     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────┐
│                      Coordinator (Facade)                        │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────┐ │
│  │ ClusterMgr  │ │ InferenceEng │ │ HealthMgr  │ │ Metrics   │ │
│  └──────┬──────┘ └──────┬───────┘ └─────┬──────┘ └─────┬─────┘ │
│         │               │               │              │       │
│         v               v               v              v       │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────┐ │
│  │ NodeRegistar│ │ TokenGen     │ │ Straggler  │ │ Latency   │ │
│  │ ModelRouter │ │ BatchSched   │ │ Recovery   │ │ Tracker   │ │
│  └─────────────┘ └──────────────┘ └────────────┘ └───────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────┐
│                   Pipeline Orchestrator                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│  │ Strategy     │ │ Tensor       │ │ Checkpoint               ││
│  │ Selector     │ │ Transport    │ │ Manager                  ││
│  └──────────────┘ └──────────────┘ └──────────────────────────┘│
│  Strategies: Sequential | Overlap | 1F1B | Staged | Disagg     │
└───────────────────────────────┬─────────────────────────────────┘
                                │ gRPC / QUIC / NCCL
                                v
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
│   Worker 0   │  │   Worker 1   │  │   Worker 2   │  │  Worker N│
│  Layers 0-15 │  │ Layers 16-31 │  │ Layers 32-47 │  │  ...     │
│  GPU: A100   │  │ GPU: RTX4090 │  │ GPU: RTX3090 │  │          │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────┘
```

---

## Data Flow

### Request Processing

```
1. API Request
   ↓
2. Auth Middleware → Rate Limit → Request ID
   ↓
3. Coordinator.generate()
   ↓
4. InferenceEngine._generate_distributed()
   ↓
5. PipelineOrchestrator.run_pipeline()
   ├─ Strategy Selection (Sequential/Overlap/Staged)
   ├─ For each node:
   │   ├─ Prepare hidden states
   │   ├─ Serialize to protobuf
   │   ├─ gRPC ForwardPass to worker
   │   ├─ Receive logits
   │   └─ Checkpoint for recovery
   └─ Return logits
   ↓
6. TokenGenerator.sample()
   ↓
7. API Response
```

### Token-by-Token Generation

```
Prompt: "Hello, how are you?"

Step 1: Encode → [15496, 11, 703, 389, 345, 30]
Step 2: Forward pass through all nodes → logits
Step 3: Sample next token → 42
Step 4: Append to sequence → [15496, 11, 703, 389, 345, 30, 42]
Step 5: Repeat steps 2-4 until EOS or max_tokens
Step 6: Decode tokens → "I'm doing well, thank you!"
```

---

## Key Design Decisions

### 1. Pipeline Parallelism over Tensor Parallelism

**Why**: Works over any network (LAN, WAN), not just NVLink.

| | Pipeline Parallelism | Tensor Parallelism |
|---|---|---|
| Network | Any (1+ Gbps) | NVLink (600+ GB/s) |
| Latency | O(N × hop_latency) | O(layer_latency) |
| Use case | Consumer GPUs, WAN | Data center, NVLink |

### 2. Coordinator-Worker Architecture

**Why**: Single point of control for scheduling, routing, and health management.

- **Coordinator**: Scheduling, topology, health, API
- **Workers**: Model execution, gRPC serving

### 3. OpenAI-Compatible API

**Why**: Zero-friction adoption for existing applications.

```python
# Same code works with OpenAI or DistLLM
client = openai.OpenAI(base_url="http://localhost:8000/v1")
```

### 4. Gossip Protocol for Cache Discovery

**Why**: Decentralized, fault-tolerant, eventually consistent.

- No single point of failure
- Self-healing after partitions
- Low bandwidth overhead

---

## Module Map

```
src/distllm/
├── api/              # FastAPI REST API
│   ├── routes/       # Endpoint handlers
│   ├── middleware.py  # Auth, rate limiting, request ID
│   └── server.py     # App factory, lifespan
├── core/             # Core inference engine
│   ├── coordinator.py      # Main facade
│   ├── inference_engine.py  # Generation modes
│   ├── batch_scheduler.py   # Continuous batching
│   ├── kv_cache.py          # KV cache management
│   ├── token_generator.py   # Sampling
│   └── model_router.py      # Content-based routing
├── dist/             # Distributed infrastructure
│   ├── pipeline/     # Pipeline orchestrator
│   ├── worker.py     # Worker node
│   ├── node_service.py  # gRPC server
│   └── p2p/          # P2P gossip, discovery
├── models/           # Model loading, partitioning
├── security/         # Auth, encryption, SSRF protection
├── observability/    # Metrics, tracing, logging
└── config/           # Configuration management
```

---

## Communication Protocols

| Protocol | Use Case | Latency | Bandwidth |
|----------|----------|---------|-----------|
| **gRPC** | Node-to-node forward pass | 1-5ms (LAN) | High |
| **QUIC** | WAN inference | 50-200ms | Medium |
| **NCCL** | Same-machine multi-GPU | <1ms | Very high |
| **HTTP/REST** | API endpoints | 5-20ms | Low |
| **WebSocket** | Real-time metrics | <10ms | Low |

---

## State Management

| State | Location | Persistence |
|-------|----------|-------------|
| Node topology | Coordinator (in-memory) | Replicated via etcd |
| KV cache | Worker GPUs | Checkpointed to disk |
| Request state | Coordinator | Thread-safe dicts |
| Health state | HealthManager | In-memory with history |
| Metrics | MetricsCollector | Prometheus export |

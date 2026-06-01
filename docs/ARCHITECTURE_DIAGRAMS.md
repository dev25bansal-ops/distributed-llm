# Architecture Diagrams

Mermaid diagrams for DistLLM's internal architecture. These complement
the textual description in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Server
    participant Auth as Auth Middleware
    participant Router as Model Router
    participant Coord as Coordinator
    participant Sched as Batch Scheduler
    participant Pipeline as Pipeline Orchestrator
    participant Node0 as Worker Node 0
    participant Node1 as Worker Node 1

    Client->>API: POST /v1/chat/completions
    API->>Auth: Validate API key
    Auth->>Router: Resolve model
    Router->>Coord: generate(prompt, params)
    Coord->>Sched: Add to queue
    Sched->>Pipeline: Execute pipeline
    
    Pipeline->>Node0: ForwardPass (layers 0-15)
    Node0-->>Pipeline: Hidden states + KV cache
    Pipeline->>Node1: ForwardPass (layers 16-31)
    Node1-->>Pipeline: Logits
    
    Pipeline-->>Coord: Generated tokens
    Coord-->>API: Response
    API-->>Client: SSE stream / JSON response
```

## Coordinator Architecture

```mermaid
graph TB
    subgraph Coordinator
        CM[Cluster Manager]
        IE[Inference Engine]
        HM[Health Manager]
        MC[Metrics Collector]
        BS[Batch Scheduler]
        MR[Model Router]
    end

    subgraph Pipeline
        PO[Pipeline Orchestrator]
        TS[Tensor Transport]
        SD[Speculative Decoder]
    end

    subgraph Workers
        N0[Node 0<br/>Layers 0-15]
        N1[Node 1<br/>Layers 16-31]
        N2[Node 2<br/>Layers 32-47]
    end

    CM --> N0
    CM --> N1
    CM --> N2
    IE --> PO
    PO --> TS
    TS --> N0
    N0 --> N1
    N1 --> N2
    HM -.->|health probes| N0
    HM -.->|health probes| N1
    HM -.->|health probes| N2
```

## KV Cache Hierarchy

```mermaid
graph LR
    subgraph L1_GPU[GPU Memory]
        PA[PagedAttention<br/>Blocks]
        PC[Prefix Cache<br/>Radix Tree]
    end

    subgraph L2_CPU[CPU RAM]
        HC[Hybrid Cache<br/>Swap]
        SC[Semantic Cache]
    end

    subgraph L3_Redis[Redis]
        RP[Redis Prompt<br/>Cache]
    end

    PA -->|evict| HC
    HC -->|evict| RP
    PC -->|lookup| SC
    SC -->|miss| RP
```

## Speculative Decoding Flow

```mermaid
sequenceDiagram
    participant Target as Target Model (GPU)
    participant Draft as Draft Model (CPU)
    participant Verifier

    Target->>Draft: Generate K candidates
    Draft-->>Verifier: Draft tokens + logprobs
    Verifier->>Target: Verify all K in one forward pass
    Target-->>Verifier: Target logits
    
    alt Accept N tokens
        Verifier->>Verifier: Accept N matching tokens
    else Reject
        Verifier->>Target: Sample correction token
    end
    
    Note over Verifier: Adaptive K based on<br/>acceptance rate
```

## Multi-Cloud Routing

```mermaid
graph TB
    subgraph Router[Unified Router]
        CR[Cross-Cloud Router]
        MP[Marketplace]
        AR[Arbitrage Engine]
    end

    subgraph Cloud[Cloud Providers]
        AWS[AWS<br/>p4d.24xlarge]
        GCP[GCP<br/>a2-highgpu]
        AZ[Azure<br/>NC48ads_A100]
    end

    subgraph Peer[Peer Network]
        P1[GPU Peer A]
        P2[GPU Peer B]
    end

    CR --> AWS
    CR --> GCP
    CR --> AZ
    MP --> P1
    MP --> P2
    AR -->|cost-optimize| CR
    AR -->|carbon-optimize| CR
```

## Federation Architecture

```mermaid
graph TB
    subgraph ClusterA[Cluster A]
        CA[Coordinator A]
        WA1[Worker A1]
        WA2[Worker A2]
    end

    subgraph ClusterB[Cluster B]
        CB[Coordinator B]
        WB1[Worker B1]
    end

    subgraph Federation
        FD[Federation Discovery]
        FR[Federation Router]
        CS[Cache Sync]
    end

    CA <-->|heartbeat| FD
    CB <-->|heartbeat| FD
    FD --> FR
    FR -->|spillover| CB
    CA <-->|KV cache digest| CS
    CB <-->|KV cache digest| CS
```

## Security Layers

```mermaid
graph TB
    subgraph External[External Request]
        REQ[HTTP/gRPC Request]
    end

    subgraph Security[Security Layers]
        TLS[TLS Termination]
        CORS[CORS Validation]
        AUTH[API Key Auth]
        RBAC[Role-Based Access]
        RATE[Rate Limiting]
        SSRF[SSRF Protection]
        AUDIT[Audit Logging]
    end

    subgraph Internal[Internal]
        APP[Application]
    end

    REQ --> TLS
    TLS --> CORS
    CORS --> AUTH
    AUTH --> RBAC
    RBAC --> RATE
    RATE --> SSRF
    SSRF --> AUDIT
    AUDIT --> APP
```

## State Machine: Node Lifecycle

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> FOLLOWER: Join cluster
    FOLLOWER --> CANDIDATE: Leader timeout
    CANDIDATE --> LEADER: Win election
    CANDIDATE --> FOLLOWER: Lose election
    LEADER --> FOLLOWER: Higher-ID peer joins
    LEADER --> RECOVERING: Node failure
    RECOVERING --> LEADER: Recovery complete
    FOLLOWER --> SHUTDOWN: Graceful stop
    LEADER --> SHUTDOWN: Graceful stop
    RECOVERING --> SHUTDOWN: Force stop
    SHUTDOWN --> [*]
```

## State Machine: Node Health

```mermaid
stateDiagram-v2
    [*] --> OFFLINE
    OFFLINE --> DEGRADED: Probe succeeds
    DEGRADED --> HEALTHY: N consecutive successes
    HEALTHY --> DEGRADED: Latency spike
    DEGRADED --> UNHEALTHY: N consecutive failures
    UNHEALTHY --> OFFLINE: Prolonged failure
    UNHEALTHY --> DEGRADED: Probe succeeds
    OFFLINE --> DRAINING: Manual drain
    DRAINING --> OFFLINE: Drain complete
```

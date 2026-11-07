# Scaling Guide — From 2 Nodes to 100

## Scaling Stages

```
Stage 1: 2-4 nodes    → Single LAN, basic setup
Stage 2: 5-16 nodes   → Multiple LANs, federation
Stage 3: 17-50 nodes  → Multi-region, autoscaling
Stage 4: 50+ nodes    → Global distribution, marketplace
```

---

## Stage 1: Small Cluster (2-4 Nodes)

### Setup

```bash
# Coordinator
distllm system coordinator --model llama-3-70b --port 50050

# Workers (on same LAN)
distllm system run --coordinator-host 192.168.1.100 --port 50051
distllm system run --coordinator-host 192.168.1.100 --port 50052
distllm system run --coordinator-host 192.168.1.100 --port 50053
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `distllm system run`
> (src/distllm/cli/main.py:357) accepts --model/--local/--config/--port/
> --dtype/--max-tokens/--temperature/--prompt/--debug only; it has no
> --coordinator-host option. Worker nodes are started with the
> `distllm-node` console script (distllm.dist.worker:main), which takes
> --node-id/--model/--start-layer/--end-layer/--total-layers/--port/
> --coordinator-host/--coordinator-port (src/distllm/config/resolver.py:82).

### Optimization

```bash
# Enable overlap pipeline (2-4x throughput)
# config.yaml: hybrid_parallel.pp_overlap: true (default true)
# Env var: DISTLLM__HYBRID_PARALLEL__PP_OVERLAP=true

# Enable continuous batching
# config.yaml: batching.max_batch_size: 32 (default)
# Env var: DISTLLM__BATCHING__MAX_BATCH_SIZE=8
```

### Monitoring

```bash
# Check cluster status
distllm cluster status

# View metrics
distllm system observe
```

---

## Stage 2: Medium Cluster (5-16 Nodes)

### Federation Setup

```bash
# Cluster A (US East)
distllm system coordinator --model llama-3-70b \
  --federate --federation-cluster-id us-east \
  --federation-seed 10.0.0.1:50060

# Cluster B (US West)
distllm system coordinator --model llama-3-70b \
  --federate --federation-cluster-id us-west \
  --federation-seed 10.0.0.1:50060
```

### Load Balancing

```yaml
# config.yaml
load_balancer:
  strategy: "least_connections"  # or "round_robin", "weighted"
  health_check_interval: 10
  max_connections_per_node: 32
```

### Auto-Scaling

The AutoScaler in src/distllm/dist/autoscaler.py is config-driven
(min_workers, max_workers, scale_up_threshold, scale_down_threshold,
cooldown_seconds, poll_interval_seconds constructor args); the coordinator
reads the poll interval from DISTLLM_AUTOSCALER_INTERVAL_S
(src/distllm/core/coordinator.py:1206). There is no DISTLLM_AUTOSCALE_* env var.

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
# Enable autoscaling
export DISTLLM_AUTOSCALE_ENABLED=true
export DISTLLM_AUTOSCALE_MIN_NODES=4
export DISTLLM_AUTOSCALE_MAX_NODES=16
export DISTLLM_AUTOSCALE_TARGET_UTILIZATION=70
```

---

## Stage 3: Large Cluster (17-50 Nodes)

### Kubernetes Deployment

```yaml
# deploy/kustomize/production/kustomization.yaml
resources:
  - ../base
patches:
  - target:
      kind: Deployment
      name: distllm-coordinator
    patch: |
      - op: replace
        path: /spec/replicas
        value: 3  # HA coordinators
  - target:
      kind: StatefulSet
      name: distllm-worker
    patch: |
      - op: replace
        path: /spec/replicas
        value: 20
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: distllm-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: distllm-worker
  minReplicas: 4
  maxReplicas: 50
  metrics:
    - type: Pods
      pods:
        metric:
          name: distllm_coordinator_queue_depth
        target:
          type: AverageValue
          averageValue: "10"
```

### Multi-Region

```bash
# Region 1: US East
kubectl config use-context us-east
kubectl apply -k deploy/kustomize/production/

# Region 2: EU West
kubectl config use-context eu-west
kubectl apply -k deploy/kustomize/production/
```

---

## Stage 4: Global Distribution (50+ Nodes)

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  Global Load Balancer            │
└───────────┬─────────────┬─────────────┬─────────┘
            │             │             │
            v             v             v
     ┌──────────┐  ┌──────────┐  ┌──────────┐
     │ US East  │  │ EU West  │  │ AP South │
     │ Cluster  │  │ Cluster  │  │ Cluster  │
     │ 20 nodes │  │ 15 nodes │  │ 10 nodes │
     └──────────┘  └──────────┘  └──────────┘
            │             │             │
            └─────────────┼─────────────┘
                          │
                          v
                  ┌──────────────┐
                  │  Federation  │
                  │  Coordinator │
                  └──────────────┘
```

### Cost Optimization

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
# Use spot instances for workers
export DISTLLM_SPOT_INSTANCES=true
export DISTLLM_SPOT_MAX_PRICE=0.50

# Enable carbon-aware routing
export DISTLLM_CARBON_AWARE=true

# Cost allocation per tenant
export DISTLLM_COST_TRACKING=true
```

Nearest real mechanisms (no env vars):
- `cost.spot_preference` in config.yaml (CostSettings, src/distllm/config/_deployment.py:70)
- Carbon-aware migration via DISTLLM_CARBON_API_KEY / DISTLLM_CARBON_THRESHOLD
  (src/distllm/core/carbon_migration.py:37,41 and src/distllm/core/coordinator.py:1409)

### Performance at Scale

| Nodes | Expected Throughput | Latency (P99) |
|-------|-------------------|---------------|
| 4 | 20 tok/s | 500ms |
| 16 | 80 tok/s | 200ms |
| 50 | 200 tok/s | 100ms |
| 100+ | 500+ tok/s | 50ms |

(Values above are unsourced estimates, not benchmark results.)

---

## Common Scaling Issues

### Coordinator Bottleneck

**Symptom**: High CPU on coordinator, slow scheduling

**Solution**:
```bash
# Enable coordinator HA via DISTLLM_HA_SECRET (src/distllm/core/ha_coordinator.py:59)
export DISTLLM_HA_SECRET=<shared-secret>
export DISTLLM_FEDERATION_PEERS="coord-2:50050,coord-3:50050"
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — no DISTLLM_HA_ENABLED or
> DISTLLM_HA_PEERS env var exists.

### Network Saturation

**Symptom**: High latency, low throughput

**Solution**:
```bash
# QUIC transport is available in src/distllm/dist/p2p/quic_transport.py
# (auto-selected via get_optimal_transport when aioquic is installed).
# There is no DISTLLM_WAN_TRANSPORT / DISTLLM_GRPC_COMPRESSION env var.
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
# Enable QUIC for WAN
export DISTLLM_WAN_TRANSPORT=quic

# Enable compression
export DISTLLM_GRPC_COMPRESSION=gzip
```

### Memory Pressure

**Symptom**: OOM errors, KV cache evictions

**Solution**:
```bash
# Defragmentation is the defragmentation config section:
# config.yaml: defragmentation.enabled: true
# Env var: DISTLLM__DEFRAGMENTATION__ENABLED=true
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — no DISTLLM_KV_CACHE_QUANT_BITS
> or DISTLLM_DEFRAG_ENABLED env var exists. KV cache quantization has no
> config/env mechanism; defrag is the defragmentation config section.
```bash
# Enable KV cache quantization
export DISTLLM_KV_CACHE_QUANT_BITS=8

# Enable defragmentation
export DISTLLM_DEFRAG_ENABLED=true
```

### Straggler Nodes

**Symptom**: Slow nodes bottleneck the pipeline

**Solution**:
```bash
# Straggler detection is always-on via StragglerDetector
# (src/distllm/core/coordinator.py:132); threshold comes from the
# straggler_threshold config field (default 1.5,
# src/distllm/config/_parallelism.py:115), not an env var.
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
# Enable straggler detection
export DISTLLM_STRAGGLER_DETECTION=true
export DISTLLM_STRAGGLER_THRESHOLD=2.0
```

---

## Monitoring at Scale

### Key Metrics

Real metric names (Prometheus, src/distllm/observability/exporter.py):

```bash
# Cluster throughput
distllm_tokens_per_second

# Per-node latency
distllm_node_latency_p99_ms

# Queue depth
distllm_coordinator_queue_depth

# Error rate
distllm_errors_total
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the names below do not exist:
```bash
# Cluster throughput
distllm_metrics_tokens_per_second_total

# Per-node latency
distllm_metrics_node_latency_seconds

# Queue depth
distllm_metrics_queue_depth

# Error rate
distllm_metrics_error_rate
```

### Grafana Dashboards

- **Cluster Overview**: Total throughput, latency, error rate
- **Node Health**: Per-node GPU, memory, latency
- **Cost Tracking**: Per-user cost, budget utilization
- **Federation Health**: Cross-cluster latency, sync status

---

## Checklist for Production

- [ ] TLS enabled on all connections
- [ ] API keys configured (not default)
- [ ] Rate limiting enabled
- [ ] Monitoring configured (Prometheus + Grafana)
- [ ] Alerting configured (PagerDuty, Slack)
- [ ] Backup/restore tested
- [ ] Runbook documented
- [ ] Load testing completed
- [ ] Security audit completed
- [ ] Disaster recovery tested

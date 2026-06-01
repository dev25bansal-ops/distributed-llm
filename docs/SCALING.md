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

### Optimization

```bash
# Enable overlap pipeline (2-4x throughput)
export DISTLLM_ENABLE_PIPELINE_OVERLAP=true

# Enable continuous batching
export DISTLLM_MAX_BATCH_SIZE=8
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
# kustomize/production/kustomization.yaml
resources:
  - ../../deploy/inference
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
          name: distllm_queue_depth
        target:
          type: AverageValue
          averageValue: "10"
```

### Multi-Region

```bash
# Region 1: US East
kubectl config use-context us-east
kubectl apply -k kustomize/production/

# Region 2: EU West
kubectl config use-context eu-west
kubectl apply -k kustomize/production/
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

```bash
# Use spot instances for workers
export DISTLLM_SPOT_INSTANCES=true
export DISTLLM_SPOT_MAX_PRICE=0.50

# Enable carbon-aware routing
export DISTLLM_CARBON_AWARE=true

# Cost allocation per tenant
export DISTLLM_COST_TRACKING=true
```

### Performance at Scale

| Nodes | Expected Throughput | Latency (P99) |
|-------|-------------------|---------------|
| 4 | 20 tok/s | 500ms |
| 16 | 80 tok/s | 200ms |
| 50 | 200 tok/s | 100ms |
| 100+ | 500+ tok/s | 50ms |

---

## Common Scaling Issues

### Coordinator Bottleneck

**Symptom**: High CPU on coordinator, slow scheduling

**Solution**:
```bash
# Enable coordinator HA
export DISTLLM_HA_ENABLED=true
export DISTLLM_HA_PEERS="coord-2:50050,coord-3:50050"
```

### Network Saturation

**Symptom**: High latency, low throughput

**Solution**:
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
# Enable KV cache quantization
export DISTLLM_KV_CACHE_QUANT_BITS=8

# Enable defragmentation
export DISTLLM_DEFRAG_ENABLED=true
```

### Straggler Nodes

**Symptom**: Slow nodes bottleneck the pipeline

**Solution**:
```bash
# Enable straggler detection
export DISTLLM_STRAGGLER_DETECTION=true
export DISTLLM_STRAGGLER_THRESHOLD=2.0
```

---

## Monitoring at Scale

### Key Metrics

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

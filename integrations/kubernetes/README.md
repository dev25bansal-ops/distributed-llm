# Kubernetes Deployment for DistLLM

Deploy DistLLM on Kubernetes using Helm charts or the Python operator.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  DistLLM        │     │  DistLLM        │
│  Coordinator    │─────│  Workers        │
│  (REST API)     │     │  (GPU Pods)     │
└────────┬────────┘     └─────────────────┘
         │
    ┌────▼────┐
    │  Redis  │
    │ (Cache) │
    └─────────┘
```

## Quick Start

### Prerequisites

- Kubernetes 1.19+
- Helm 3.8+
- NVIDIA GPU Operator (for GPU scheduling)

### Install via Helm

```bash
# Add repo and install
helm install distllm ./integrations/kubernetes/helm/distllm \
  --set image.repository=your-registry/distributed-llm \
  --set image.tag=latest \
  --set resources.limits.nvidia.com/gpu=1
```

### Install the Operator

```bash
# 1. Create the CRD
kubectl apply -f integrations/kubernetes/operator/distllm_operator/crd.yaml

# 2. Build and deploy the operator
docker build -f integrations/kubernetes/operator/Dockerfile.operator -t distllm-operator:latest .
kubectl apply -f deploy/operator.yaml

# 3. Create a DistLLMCluster resource
cat <<EOF | kubectl apply -f -
apiVersion: distllm.ai/v1
kind: DistLLMCluster
metadata:
  name: my-cluster
spec:
  replicas: 3
  model: "meta-llama/Llama-2-7b"
EOF
```

## Configuration

### Helm Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replicaCount` | `1` | Number of coordinator replicas |
| `image.repository` | `distributed-llm` | Docker image |
| `resources.limits.nvidia.com/gpu` | `1` | GPU limit per worker |
| `redis.enabled` | `true` | Enable Redis cache |
| `autoscaling.enabled` | `false` | Enable HPA |

### Custom Resource Spec

```yaml
apiVersion: distllm.ai/v1
kind: DistLLMCluster
metadata:
  name: production
spec:
  replicas: 4
  model: "meta-llama/Llama-2-70b"
  resources:
    cpu: "8"
    memory: "32Gi"
    gpu: "2"
```

## Production Deployment

```bash
# With Redis, HPA, and ServiceMonitor
helm install distllm-prod ./integrations/kubernetes/helm/distllm \
  --set replicaCount=3 \
  --set redis.enabled=true \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=3 \
  --set autoscaling.maxReplicas=10 \
  --set serviceMonitor.enabled=true
```

## Uninstall

```bash
helm uninstall distllm
kubectl delete crd distllmclusters.distllm.ai
```

# Deployment Guide

Production deployment guide for Distributed LLM. Covers local, Docker, Docker Compose, and Kubernetes deployments.

## Table of Contents

- [Configuration](#configuration)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Docker Compose Multi-Node](#docker-compose-multi-node)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Security Checklist](#security-checklist)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Configuration

Configuration follows this precedence: **CLI args > environment variables > config.yaml > defaults**.

### config.yaml

```yaml
model:
  name: "HuggingFaceTB/SmolLM-135M"
  dtype: "float16"

coordinator:
  host: "localhost"
  port: 50050
  api_port: 8000

nodes:
  - node_id: "node_0"
    host: "localhost"
    port: 50051
    start_layer: 0
    end_layer: 3

  - node_id: "node_1"
    host: "localhost"
    port: 50052
    start_layer: 4
    end_layer: 7

generation:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9

logging:
  level: "INFO"
  format: "json"

network:
  grpc_timeout: 30
  max_retries: 3
  retry_delay: 1.0

tls:
  enabled: false
  cert_dir: "certs"
```

### Environment Variables

```bash
# Core
DISTLLM__MODEL__NAME=HuggingFaceTB/SmolLM-135M
DISTLLM__MODEL__DTYPE=float16
DISTLLM__COORDINATOR__HOST=0.0.0.0
DISTLLM__COORDINATOR__API_PORT=8000

# Security
API_KEY=your-secret-key-here
DISTLLM__TLS__ENABLED=true
DISTLLM__CORS_ORIGINS=https://yourdomain.com

# Performance
DISTLLM__BATCHING__MAX_BATCH_SIZE=32
DISTLLM__PREFIX_CACHE__ENABLED=true
DISTLLM__RATE_LIMIT__ENABLED=true
DISTLLM__RATE_LIMIT__DEFAULT_RPM=60

# Profiles (dev / staging / production)
DISTLLM__PROFILE=production
```

### Validate Configuration

```bash
distllm validate-config
# or
python -m distllm.api.server --model test --validate-config
```

## Local Development

### Single Machine (Local Mode)

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"

# Start API server with local model
distllm api --model meta-llama/Llama-3.2-1B --local --port 8000

# Or use config file
distllm run --model meta-llama/Llama-3.2-1B --local --config config.yaml --port 8000

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

### Interactive Chat

```bash
distllm chat --model meta-llama/Llama-3.2-1B --local
```

## Docker Deployment

### Build Image

```bash
docker build -t distllm:latest .
```

### Run Container

```bash
docker run -d \
  --gpus all \
  --name distllm \
  -p 8000:8000 \
  -p 50051:50051 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e API_KEY=your-secret-key \
  -e DISTLLM__MODEL__NAME=meta-llama/Llama-3.2-1B \
  distllm:latest \
  distllm-api --model meta-llama/Llama-3.2-1B --local --config /app/config.yaml
```

### Non-Root Execution

The Dockerfile runs as non-root user `distllm` by default. For host-mounted volumes:

```bash
docker run -d \
  --gpus all \
  --user 1000:1000 \
  --name distllm \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e DISTLLM__MODEL__NAME=/models/my-model \
  distllm:latest
```

## Docker Compose Multi-Node

### Coordinator + 2 Workers

```yaml
# docker-compose.yml
version: "3.8"

services:
  coordinator:
    build: .
    ports:
      - "8000:8000"
      - "50050:50050"
    environment:
      - API_KEY=${API_KEY}
      - DISTLLM__MODEL__NAME=${MODEL_NAME}
      - DISTLLM__TLS__ENABLED=${TLS_ENABLED:-false}
    volumes:
      - ./config.yaml:/app/config.yaml
    command: >
      distllm-coordinator --model ${MODEL_NAME}
      --nodes worker1:50051:0:15,worker2:50052:16:31
      --total-layers 32
    depends_on:
      - worker1
      - worker2

  worker1:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    environment:
      - DISTLLM__MODEL__NAME=${MODEL_NAME}
    volumes:
      - hf-cache:/root/.cache/huggingface
    command: >
      distllm-node --node-id worker1
      --model ${MODEL_NAME}
      --start-layer 0 --end-layer 15 --total-layers 32
      --coordinator-host coordinator --coordinator-port 50050

  worker2:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    environment:
      - DISTLLM__MODEL__NAME=${MODEL_NAME}
    volumes:
      - hf-cache:/root/.cache/huggingface
    command: >
      distllm-node --node-id worker2
      --model ${MODEL_NAME}
      --start-layer 16 --end-layer 31 --total-layers 32
      --coordinator-host coordinator --coordinator-port 50050

volumes:
  hf-cache:
```

```bash
# Start
docker-compose up -d

# Check health
curl http://localhost:8000/health
curl http://localhost:8000/ready

# View logs
docker-compose logs -f coordinator
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+ with GPU nodes (NVIDIA device plugin)
- kubectl configured
- Helm (optional, for monitoring stack)

### Namespace and Secrets

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: distllm
```

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: distllm-secrets
  namespace: distllm
type: Opaque
stringData:
  api-key: "your-secret-key"
  hf-token: "your-huggingface-token"  # if needed
```

### Coordinator Deployment

```yaml
# coordinator-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distllm-coordinator
  namespace: distllm
  labels:
    app: distllm-coordinator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: distllm-coordinator
  template:
    metadata:
      labels:
        app: distllm-coordinator
    spec:
      containers:
        - name: coordinator
          image: distllm:latest
          args:
            - "distllm-api"
            - "--model"
            - "$(MODEL_NAME)"
            - "--local"
            - "--config"
            - "/app/config.yaml"
          env:
            - name: MODEL_NAME
              value: "meta-llama/Llama-3.2-1B"
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: distllm-secrets
                  key: api-key
            - name: DISTLLM__BATCHING__MAX_BATCH_SIZE
              value: "32"
            - name: DISTLLM__PREFIX_CACHE__ENABLED
              value: "true"
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 50050
              name: grpc
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
          livenessProbe:
            httpGet:
              path: /live
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
          volumeMounts:
            - name: config
              mountPath: /app/config.yaml
              subPath: config.yaml
      volumes:
        - name: config
          configMap:
            name: distllm-config
```

### Worker Node Deployment

```yaml
# worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distllm-worker-0
  namespace: distllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: distllm-worker
      node-id: worker-0
  template:
    metadata:
      labels:
        app: distllm-worker
        node-id: worker-0
    spec:
      containers:
        - name: worker
          image: distllm:latest
          args:
            - "distllm-node"
            - "--node-id"
            - "worker-0"
            - "--model"
            - "$(MODEL_NAME)"
            - "--start-layer"
            - "0"
            - "--end-layer"
            - "15"
            - "--total-layers"
            - "32"
            - "--coordinator-host"
            - "distllm-coordinator"
            - "--coordinator-port"
            - "50050"
          env:
            - name: MODEL_NAME
              value: "meta-llama/Llama-3.2-1B"
          resources:
            limits:
              nvidia.com/gpu: "1"
              cpu: "4"
              memory: "16Gi"
            requests:
              nvidia.com/gpu: "1"
              cpu: "2"
              memory: "8Gi"
          volumeMounts:
            - name: hf-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: hf-cache
          emptyDir:
            sizeLimit: 50Gi
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: distllm-coordinator
  namespace: distllm
spec:
  selector:
    app: distllm-coordinator
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: grpc
      port: 50050
      targetPort: 50050
  type: ClusterIP
```

### Ingress (Optional)

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: distllm-ingress
  namespace: distllm
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  rules:
    - host: distllm.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: distllm-coordinator
                port:
                  number: 8000
```

### Apply

```bash
kubectl apply -f namespace.yaml
kubectl create -f secrets.yaml
kubectl create configmap distllm-config --from-file=config.yaml -n distllm
kubectl apply -f coordinator-deployment.yaml
kubectl apply -f worker-deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml  # optional

# Check status
kubectl get pods -n distllm
kubectl get svc -n distllm
kubectl describe pod -l app=distllm-coordinator -n distllm
```

## Security Checklist

### Before Production Deployment

- [ ] Set `API_KEY` environment variable
- [ ] Enable TLS (`DISTLLM__TLS__ENABLED=true`) with valid certificates
- [ ] Configure CORS origins (`DISTLLM__CORS_ORIGINS`)
- [ ] Enable rate limiting (`DISTLLM__RATE_LIMIT__ENABLED=true`)
- [ ] Use non-root Docker container (default)
- [ ] Set appropriate file permissions on config files
- [ ] Use Kubernetes secrets or Vault for sensitive values
- [ ] Configure network policies to restrict pod-to-pod communication
- [ ] Enable audit logging
- [ ] Set resource limits (CPU, memory, GPU)

### TLS Setup

```bash
# Generate self-signed certs (development only)
openssl req -x509 -newkey rsa:4096 \
  -keyout server.key -out server.crt \
  -days 365 -nodes \
  -subj "/CN=distributed-llm" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,DNS:distllm,DNS:*.local"

# For production: use Let's Encrypt or your CA
```

```yaml
# config.yaml
tls:
  enabled: true
  cert_dir: "/app/certs"
  cert_file: "/app/certs/server.crt"
  key_file: "/app/certs/server.key"
  ca_cert_file: "/app/certs/ca.crt"
```

## Monitoring

### Health Endpoints

| Endpoint | Purpose | Returns 200 when |
|---|---|---|
| `/health` | General health | Model loaded, nodes responsive |
| `/ready` | Kubernetes readiness | Can accept traffic |
| `/live` | Kubernetes liveness | Process alive, not deadlocked |
| `/metrics` | Prometheus metrics | Always returns metrics |

### Prometheus Integration

```yaml
# deploy/monitoring/prometheus/config.yml
scrape_configs:
  - job_name: distllm
    metrics_path: /metrics
    static_configs:
      - targets: ["distllm-coordinator:8000"]
```

### Key Metrics

- `distllm_service_up` - Service status (0/1)
- `distllm_coordinator_loaded` - Coordinator loaded (0/1)
- `distllm_ready` - Readiness status (0/1)
- `distllm_active_requests` - Currently processing
- `distllm_pending_requests` - In queue
- `distllm_cpu_percent` - CPU utilization
- `distllm_gpu_memory_percent` - GPU memory usage
- `distllm_gpu_temperature_c` - GPU temperature

### Grafana Dashboards

Pre-built dashboards in `deploy/monitoring/grafana/`:
- `coordinator.json` - Coordinator metrics
- `node-pool.json` - Worker node pool overview

## Troubleshooting

### Common Issues

**Node not connecting to coordinator:**
```bash
# Check network connectivity
kubectl exec -it distllm-worker-0 -n distllm -- nc -zv distllm-coordinator 50050

# Check coordinator logs
kubectl logs -l app=distllm-coordinator -n distllm
```

**503 Service Unavailable:**
- Check `/ready` endpoint for reason
- Verify nodes are healthy via `/health`
- Check backpressure: too many pending requests

**TLS errors:**
- Verify certificate SANs include all hostnames
- Check certificate expiration
- Ensure ca_cert matches server cert

**OOM errors:**
- Reduce `max_batch_size` and `max_tokens_per_batch`
- Enable prefix cache to reduce memory duplication
- Consider quantization (`quantization.method: "bitsandbytes"`)

**Slow generation:**
- Network is likely bottleneck - use 10GbE+ for distributed mode
- Reduce number of nodes if possible
- Enable speculative decoding for throughput

### Debug Mode

```bash
# Enable debug logging
distllm api --model my-model --local --debug

# Or via env var
export DISTLLM__DEBUG=true
```

# Self-Hosted Deployment Guide

Deploy open-source LLMs on your own GPUs with DistLLM's self-hosted inference backend.

## Prerequisites

- NVIDIA GPU with CUDA 12.1+ (or AMD ROCm / Apple Silicon)
- Python 3.10+
- 16GB+ VRAM recommended for 7B models, 80GB+ for 70B models

## Installation

```bash
# Minimum (routing-only, no self-hosted inference)
pip install distllm

# With self-hosted inference support
pip install "distllm[self-hosted]"

# Full install (development + all backends)
pip install "distllm[all]"
```

## Quick Start (Self-Hosted)

```bash
# Install with vLLM backend
pip install "distllm[self-hosted]"

# Start with local model (no auth for development)
distllm system api --model meta-llama/Llama-3.2-1B --local --port 8000 --no-auth

# Or with API key
export API_KEY="your-secret-key"
distllm system api --model meta-llama/Llama-3.2-1B --local --port 8000

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"model":"default"}'
```

## Configuration

Create `config.yaml`:

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  dtype: "float16"

coordinator:
  host: "0.0.0.0"
  port: 8000

vllm:
  enabled: true
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9

providers:
  openai:
    enabled: true
  together:
    enabled: true
```

## Provider Routing with Self-Hosted

DistLLM can route between your self-hosted model and cloud providers:

```yaml
routing:
  strategy: "cost"  # auto-pick cheapest

providers:
  openai:
    enabled: true
    models: ["gpt-4o-mini"]
  self-hosted:
    enabled: true
    models: ["default"]  # your local model
```

Requests to your API will automatically use the cheapest option.

## Docker

```bash
docker-compose up
```

The docker-compose starts coordinator + 2 worker nodes + Prometheus + Grafana.

## Kubernetes

```bash
# Install with Helm
helm repo add distllm https://charts.distllm.ai
helm install distllm distllm/distllm \
  --set model.name=meta-llama/Llama-3.2-1B \
  --set coordinator.replicas=1
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `gpu_memory_utilization` or use smaller model |
| `ModuleNotFoundError: torch` | Install with `pip install distllm[self-hosted]` |
| Model not loading | Add `--trust-remote-code` for architectures that need it |

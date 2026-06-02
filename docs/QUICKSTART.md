# Quick Start Guide — 5 Minutes to Distributed LLM Inference

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA 12+ (or CPU-only mode)
- 8GB+ RAM per node

---

## Option 1: Docker Compose (Recommended)

### Step 1: Clone and Configure

```bash
git clone https://github.com/distributed-llm/distributed-llm.git
cd distributed-llm

# Copy example config
cp config.yaml.example config.yaml
```

### Step 2: Start the Cluster

```bash
# Start coordinator + 2 workers
docker-compose up -d

# Check status
docker-compose ps
```

### Step 3: Download a Model

```bash
# Download a small model for testing
docker compose exec coordinator distllm model load meta-llama/Llama-3.2-1B
```

### Step 4: Chat!

```bash
# Interactive chat
docker compose exec coordinator distllm chat

# Or via API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Option 2: Python Package

### Step 1: Install

```bash
pip install distllm
```

### Step 2: Start Coordinator

```bash
# Start with a small model (no auth required for local development)
distllm system api --model meta-llama/Llama-3.2-1B --local --no-auth
```

### Step 3: Chat

```bash
distllm chat
```

---

## Option 3: Desktop App

### Step 1: Download

Download the desktop app from [releases](https://github.com/distributed-llm/distributed-llm/releases).

### Step 2: Install and Run

1. Open the app
2. Click "Download Model"
3. Select a model (e.g., Llama-3.2-1B)
4. Click "Start Chat"

---

## Authentication

### Development (No Auth)

For local development, disable authentication:

```bash
distllm system api --model meta-llama/Llama-3.2-1B --local --no-auth
```

No API key needed — anyone on localhost can connect.

### With API Key

Set an API key via environment variable:

```bash
# Set your API key
export API_KEY="your-secret-key"

# Start the server
distllm system api --model meta-llama/Llama-3.2-1B --local
```

The API key is shown on startup:
```
API Key: your-secret-key
Use: curl -H 'Authorization: Bearer your-secret-key' http://localhost:8000/health
```

### Connecting External Tools

| Tool | Configuration |
|------|--------------|
| **OpenAI SDK** | `OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")` |
| **OpenCode** | Set `base_url: http://localhost:8000/v1` |
| **Cursor** | Set `openai.baseUrl` in settings |
| **curl** | `curl -H "Authorization: Bearer your-key" http://localhost:8000/v1/...` |

---

## Multi-Node Setup

### Same Network (LAN)

```bash
# Node 1 (Coordinator)
$env:API_KEY = "my-cluster-key"
distllm system api --model meta-llama/Llama-3.2-70B --local --host 0.0.0.0 --port 8000

# Node 2 (Worker)
$env:API_KEY = "my-cluster-key"
distllm cluster join --coordinator 192.168.1.100 --port 50050

# Node 3 (Worker)
$env:API_KEY = "my-cluster-key"
distllm cluster join --coordinator 192.168.1.100 --port 50051
```

**Important**: All nodes must use the same `API_KEY`.

### Different Locations (WAN)

```bash
# Enable WAN mode
export DISTLLM_WAN_ENABLED=true
export DISTLLM_WAN_TRANSPORT=quic

# Start coordinator
distllm system coordinator --model meta-llama/Llama-3.2-70B

# Workers connect from anywhere
distllm system run --coordinator-host your-server.com --port 50050
```

---

## Using the API

### Python SDK

```python
from distllm_sdk import DistLLMClient

client = DistLLMClient(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",  # or "not-needed" with --no-auth
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "Explain distributed computing"}],
)
print(response.choices[0].message.content)
```

### OpenAI SDK (Drop-in Replacement)

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",  # or "not-needed" with --no-auth
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

---

## Next Steps

- **Add more nodes**: See [Scaling Guide](SCALING.md)
- **Optimize performance**: See [Performance Tuning](PERFORMANCE_TUNING.md)
- **Secure your cluster**: See [Security Hardening](SECURITY_HARDENING.md)
- **Deploy to production**: See [Deployment Guide](../DEPLOYMENT.md)
- **Write plugins**: See [Plugin Development](PLUGIN_DEVELOPMENT.md)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Use a smaller model or enable quantization |
| `Connection refused` | Check firewall, verify coordinator is running |
| `Model not found` | Run `distllm model load <model-name>` first |
| `Unauthorized` | Set `API_KEY` env var or use `--no-auth` flag |
| Slow inference | Check GPU utilization with `nvidia-smi` |

See [Troubleshooting Guide](TROUBLESHOOTING.md) for more.

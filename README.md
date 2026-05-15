# Distributed LLM

**Run large language models across multiple machines. Pool GPUs. Break the memory barrier.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## What is This?

Distributed LLM lets you run models **too large for any single GPU** by splitting them across multiple machines. Use pipeline parallelism over gRPC to turn a cluster of consumer GPUs into an inference engine for 70B+ parameter models.

```
┌─────────────────┐
│   Your App      │
│   (HTTP/gRPC)   │
└────────┬────────┘
         │
┌────────▼────────┐
│   Coordinator   │  ─ Orchestrates inference, samples tokens
└────────┬────────┘
         │ gRPC (TLS optional)
    ┌────┼────┬──────────┐
    ▼    ▼    ▼          ▼
┌──────┐┌──────┐┌──────┐┌──────┐
│Node 0││Node 1││Node 2││Node N│  ─ Each runs a subset of model layers
│GPU:0 ││GPU:1 ││GPU:2 ││GPU:N │
└──────┘└──────┘└──────┘└──────┘
```

### Why Not vLLM, TGI, or Ollama?

Those solve **single-machine** inference brilliantly. This solves a different problem: **running models that don't fit on any single machine's GPU memory**.

| | vLLM / TGI | Distributed LLM |
|---|---|---|
| **Scope** | Single machine | Multiple machines |
| **Max model size** | Limited by single GPU | Limited by total cluster VRAM |
| **Use case** | Production serving on GPUs | Running large models on consumer hardware |
| **Network** | Not needed | gRPC over LAN/WAN |

## Quick Start

### Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install distributed-llm
```

### Local Mode (Single Machine)

```bash
# Interactive chat
distllm --model meta-llama/Llama-3.2-1B --local --chat

# REST API (OpenAI-compatible)
distllm-api --model meta-llama/Llama-3.2-1B --local --port 8000

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

### Distributed Mode (Multiple Machines)

**Machine 1** (has GPU 0):
```bash
distllm-node --node-id node_0 \
  --model meta-llama/Llama-3-8B \
  --start-layer 0 --end-layer 15 --total-layers 32 \
  --port 50051
```

**Machine 2** (has GPU 1):
```bash
distllm-node --node-id node_1 \
  --model meta-llama/Llama-3-8B \
  --start-layer 16 --end-layer 31 --total-layers 32 \
  --port 50052
```

**Machine 1** (Coordinator):
```bash
distllm-coordinator --model meta-llama/Llama-3-8B \
  --nodes <machine2_ip>:50052:16:31 \
  --total-layers 32
```

### Docker

```bash
docker-compose up
```

## Supported Models

| Architecture | Models | Status |
|---|---|---|
| **GPT-2** | gpt2, gpt2-large, gpt2-xl | Tested |
| **GPT-Neo** | EleutherAI/gpt-neo-1.3B | Tested |
| **Llama** | meta-llama/Llama-2-7B, Llama-3-8B | Supported |
| **Qwen2.5** | Qwen/Qwen2.5-7B | Tested |
| **Mistral** | mistralai/Mistral-7B | Supported |
| **Phi** | microsoft/Phi-3-mini | Tested |
| **StableLM** | stabilityai/stablelm-2-1_6b | Tested |
| **Pythia** | EleutherAI/pythia-6.9b | Tested |

## Performance

Benchmarks on 2x RTX 5060 over 1GbE LAN:

| Model | Single GPU | 2-Node Distributed | Overhead |
|---|---|---|---|
| TinyStories-1M (8 layers) | 120 tok/s | 45 tok/s | 2.7x |
| GPT-2 (12 layers) | 85 tok/s | 30 tok/s | 2.8x |

**Network is the bottleneck.** Performance scales with:
- Lower network latency (10GbE > 1GbE)
- Fewer nodes (1 round-trip per node per token)
- Larger batch sizes (future optimization)

See [`benchmarks/`](benchmarks/) for reproducible benchmark scripts.

## Architecture

### Pipeline Parallelism

Each node holds a subset of transformer layers. For each token:
1. Coordinator sends token IDs to first node
2. First node embeds tokens, runs its layers, sends activations to next node
3. Each middle node processes activations, passes to next
4. Last node applies final norm + LM head, returns logits
5. Coordinator samples next token, repeats

KV cache avoids re-processing previous tokens.

### Key Components

| Component | File | Purpose |
|---|---|---|
| ModelPartitioner | `src/models.py` | Load/split model layers |
| Coordinator | `src/coordinator.py` | Orchestrate inference |
| WorkerNode | `src/node.py` | Run layer subset |
| gRPC Layer | `src/communication.py` | Network communication |
| KV Cache | `src/kv_cache.py` | Cache key-value states |
| REST API | `src/api.py` | OpenAI-compatible API |

### Protocol Buffers

Service definitions in [`proto/node.proto`](proto/node.proto):
- `NodeService` - Worker nodes (ForwardPass, HealthCheck, GetNodeInfo)
- `CoordinatorService` - Coordinator (RegisterNode, Infer, StreamInfer)

Tensor serialization uses raw bytes (4-8x faster than float lists).

## Configuration

```yaml
# config.yaml
model:
  name: "meta-llama/Llama-3-8B"
  dtype: "float16"

nodes:
  - node_id: "node_0"
    host: "192.168.1.100"
    port: 50051
    start_layer: 0
    end_layer: 15

  - node_id: "node_1"
    host: "192.168.1.101"
    port: 50052
    start_layer: 16
    end_layer: 31

generation:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9

network:
  grpc_timeout: 30
  max_retries: 3
  retry_delay: 1.0
```

## API Reference

OpenAI-compatible endpoints on `http://localhost:8000`:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |

**Chat example:**
```json
{
  "model": "distributed-llm",
  "messages": [{"role": "user", "content": "Explain quantum computing"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

## Development

```bash
git clone https://github.com/distributed-llm/distributed-llm.git
cd distributed-llm
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
black src/

# Run benchmarks
python benchmarks/run.py --model roneneldan/TinyStories-1M
```

## Roadmap

| Phase | Feature | Status |
|---|---|---|
| **Phase 1** | Core distributed inference | Done |
| | KV cache management | Done |
| | OpenAI-compatible API | Done |
| | TLS support | Done |
| | Streaming chat/completions | Done |
| **Phase 2** | Speculative decoding | Done |
| | Tensor parallelism | Done |
| | LoRA adapter support | Done |
| | Prefix cache & chunked prefill | Done |
| | Continuous batching | Done |
| | Prometheus metrics | Done |
| **Phase 3** | pip installable package | Done |
| | Kubernetes readiness/liveness probes | Done |
| | Graceful shutdown | Done |
| | Structured error responses | Done |
| | Request timeouts & backpressure | Done |
| | YAML config loading | Done |
| | Security hardening (CORS, headers, auth) | Done |
| | Docker non-root execution | Done |
| **Phase 4** | Web dashboard | In Progress |
| | Auto-discovery of nodes | Planned |
| | Model compression pipeline | Planned |
| | P2P KV cache gossip | Planned |
| | Chaos engineering | Planned |
| | Canary deployments | Planned |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and contribution guidelines.

## License

Apache 2.0. See [LICENSE](LICENSE).

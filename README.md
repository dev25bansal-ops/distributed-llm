# DistLLM — Distributed Inference Across All Your Devices

**Pool GPUs from every device you own to run models no single machine can handle.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

You have a gaming PC with an RTX 4090. Your laptop has an RTX 4060. Your friend has a desktop with an RTX 3080. None of you can run Llama 3.1 70B alone. **Together, you can.**

DistLLM splits large language models across all your devices using pipeline parallelism. Each device runs a fraction of the model layers. Automatic discovery. Auto-partitioning. Works over LAN, WiFi, or internet.

```
Your Laptop (RTX 4060) ────┐
                            │
Your Gaming PC (RTX 4090) ──┼──► DistLLM Cluster ──► Run 70B models
                            │
Friend's PC (RTX 3080) ────┘
```

## Why DistLLM?

| Problem | Solution |
|---------|----------|
| One GPU can't run today's best models | Split across all your devices |
| Cloud inference costs thousands/month | Use the GPUs you already own |
| Data privacy concerns with cloud APIs | Your data stays on your devices |
| Slow single-device inference | Pipeline parallelism = faster generation |
| Setting up distributed systems is hard | One command to start, one to join |

## Quick Start

```bash
pip install distributed-llm[cli,self-hosted]

# On your main machine — start a cluster
distllm cluster start --model meta-llama/Llama-3.2-7B

# On every other machine — join the cluster
distllm cluster join
```

## How It Works

DistLLM uses **pipeline parallelism** — the model is split across devices by layers:

```
Device 1 (Laptop): Layers 0-5 ──→ Device 2 (Desktop): Layers 6-11 ──→ Device 3 (Friend's PC): Layers 12-17

Each device runs ~6 layers → fits in 6-8GB VRAM
Combined pool → runs models up to 70B parameters
```

Key capabilities:
- **Auto-discovery**: devices find each other on the same network automatically
- **Auto-partitioning**: automatically assigns layers based on each device's GPU
- **Node recovery**: if a device disconnects, remaining nodes take over
- **Straggler detection**: slow nodes are detected and worked around
- **WAN optimization**: token accumulation for low-latency cross-internet inference
- **Privacy-first**: keep sensitive layers on your own devices

## Installation

```bash
# Core package + CLI
pip install distributed-llm[cli]

# With PyTorch/transformers for self-hosted GPU inference (recommended)
pip install "distributed-llm[cli,self-hosted]"

# With vLLM backend (recommended for NVIDIA GPUs)
pip install "distributed-llm[self-hosted,backends]"

# Full install (all backends, integrations, SDKs)
pip install "distributed-llm[all]"

# Development
pip install -e ".[dev]"
```

## Key Features

- **Pipeline parallelism** — split any HuggingFace model across N devices
- **Auto-discovery** — mDNS/zeroconf device finding on LAN
- **12 backends** — vLLM, llama.cpp, TensorRT-LLM, ExLlamaV2, ONNX, PyTorch, Triton, NIM, MLX, WebGPU, Ollama, TGI
- **Auth plugin** — JWT authentication + RBAC role-based access control
- **Health watchdog** — continuous node health monitoring with auto circuit-breaking
- **Semantic caching** — deduplicate repeated prompts with embedding similarity
- **Token streaming** — `generate_stream()` for real-time token-by-token responses
- **Config validation** — cross-field validation catches invalid combinations at load time
- **Circuit breaker** — graduated backpressure for load shedding
- **Auto-partitioning** — hardware-aware DP solver for optimal layer assignment
- **Node recovery** — checkpoint-based recovery when nodes disconnect
- **Straggler detection** — statistical outlier detection for slow nodes
- **Dynamic rebalancing** — redistribute layers when nodes join/leave
- **P2P KV cache gossip** — CRDT-based cache sharing between nodes
- **Wide-area support** — token accumulation for internet-scale inference
- **Quantization** — 4-bit/8-bit to fit larger models on consumer GPUs
- **OpenAI-compatible API** — use any OpenAI client to send requests
- **Full observability** — Prometheus metrics, OTel tracing, structured logging
- **Interactive chat** — `distllm chat` for CLI-based interaction
- **Health endpoints** — `/healthz` (liveness) and `/readyz` (readiness) for Kubernetes probes

## CLI Commands

```bash
# Start the REST API server (local mode)
distllm system run --model meta-llama/Llama-3.2-1B --local --port 8000

# Start the API server on a cluster node
distllm-api --model meta-llama/Llama-3.2-1B --local

# Start a coordinator node with workers (modern CLI)
distllm cluster start --model meta-llama/Llama-3.2-7B

# Join a cluster as a worker node
distllm cluster join --coordinator 192.168.1.100

# System diagnostics (local environment check)
distllm cli-doctor

# Cluster diagnostics (check a running coordinator)
distllm system doctor

# Quantize / optimize model (adaptive precision optimizer)
distllm tune quantize --model meta-llama/Llama-3.2-7B

# Profile optimal batch size
distllm tune batch --model meta-llama/Llama-3.2-7B --gpus 1

# Size KV cache for expected concurrency
distllm tune cache --model meta-llama/Llama-3.2-7B

# Interactive chat
distllm chat
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Pipeline parallelism, node topology, KV cache management
- [Deployment](DEPLOYMENT.md) — Local, Docker, and multi-machine deployment
- [API Reference](docs/api.md) — OpenAI-compatible API docs

## Project Structure

```
src/distllm/
├── core/              # Coordinator, engine strategies, schedulers, HA
├── dist/              # Distributed inference engine
│   ├── pipeline/      # PipelineOrchestrator, transport, 1F1B scheduler
│   ├── partition/     # Hardware-aware auto-partitioner, quantization
│   ├── p2p/           # P2P gossip protocol, Kademlia DHT, discovery
│   ├── backends/      # Node-level backend adapters
│   ├── worker.py      # WorkerNode — per-device model subset (entry point)
│   ├── recovery.py    # Node failure recovery + checkpoint replay
│   ├── straggler.py   # Straggler detection
│   ├── rebalancer.py  # Dynamic pipeline rebalancing
│   └── wide_area.py   # WAN-optimized inference
├── api/               # OpenAI-compatible REST API (FastAPI)
├── backends/          # Inference backends: vLLM, llama.cpp, TensorRT, ONNX, MLX, Triton
├── models/            # Model loading, partitioning, hub
├── cli/               # Typer CLI (`distllm`, `distllm cluster`, `distllm tune`)
├── config/            # Pydantic settings, YAML/env loader
├── sdk/               # Python client library
├── observability/     # Prometheus metrics, OTel tracing, structured logging
└── plugins/           # Auth, caching, watchdog, and other built-in plugins
```

## Supported Model Architectures

GPT-2, GPT-Neo, Llama 2/3, Mistral, Mixtral, Qwen2.5, Phi, DeepSeek, StableLM, Pythia, Baichuan, ChatGLM, InternLM, and more via HuggingFace AutoModel.

## Roadmap

- **v0.4.x** (current): Pipeline parallelism across LAN devices, auto-partitioning via hardware-aware DP solver, health watchdog, semantic cache, RBAC auth, 12 backends, speculative decoding, mDNS auto-discovery
- **Planned**:
  - **GPU marketplace reputation system**
  - **Privacy-preserving split inference with differential privacy**
  - **Federated clusters, cross-internet NAT traversal, incentive systems** (2027)

> Full canonical project map: [`docs/_map/UNIFIED.md`](docs/_map/UNIFIED.md) — single entry point for architecture, modules, API, ops, security, and the hygiene log.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

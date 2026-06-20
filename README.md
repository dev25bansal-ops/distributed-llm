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
pip install distllm

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
# Core package
pip install distllm

# With vLLM backend (recommended for NVIDIA GPUs)
pip install "distllm[vllm]"

# With llama.cpp backend (CPU, AMD, Apple Silicon)
pip install "distllm[llamacpp]"

# Development
pip install -e ".[dev]"
```

## Key Features

- **Pipeline parallelism** — split any HuggingFace model across N devices
- **Auto-discovery** — mDNS/zeroconf device finding on LAN
- **6 backends** — vLLM, llama.cpp, TensorRT-LLM, ExLlamaV2, ONNX, PyTorch
- **Auth plugin** — JWT authentication + RBAC role-based access control
- **Health watchdog** — Continuous node health monitoring with auto circuit-breaking
- **Semantic caching** — Deduplicate repeated prompts with embedding similarity
- **Token streaming** — `generate_stream()` for real-time token-by-token responses
- **Config validation** — Cross-field validation catches invalid combinations at load time
- **Circuit breaker** — Graduated backpressure for load shedding
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
- **Auth plugin** — JWT authentication with RBAC role-based access control
- **Health watchdog** — continuous node health monitoring with automatic failover
- **Semantic cache** — caching plugin with deduplication for repeated prompts
- **Token streaming** — `generate_stream()` SDK method for token-by-token responses
- **Health endpoints** — `/healthz` (liveness) and `/readyz` (readiness) for Kubernetes probes

## CLI Commands

```bash
# Start a coordinator node
distllm-coordinator --model meta-llama/Llama-3.2-1B --local --chat

# Start distributed coordinator
distllm-coordinator --model meta-llama/Llama-3.2-7B \
  --nodes laptop:50051:0:5 desktop:50052:6:11 friend:50053:12:17

# Start a worker node
distllm-node --node-id laptop --model meta-llama/Llama-3.2-7B \
  --start-layer 0 --end-layer 5 --total-layers 18 \
  --coordinator-host 192.168.1.100 --coordinator-port 50050

# Start the REST API server
distllm-api --model meta-llama/Llama-3.2-1B --local

# System diagnostics (check Python, CUDA, GPU, network, ports)
distllm doctor

# Quantize a model for smaller footprint
distllm tune quantize --model meta-llama/Llama-3.2-7B --bits 4

# Batch inference over a dataset
distllm tune batch --input prompts.jsonl --output results.jsonl

# Warm the semantic cache from prior prompts
distllm tune cache --preload cache_seed.jsonl
```

## Documentation

- [Architecture](docs/architecture.md) — Pipeline parallelism, node topology, KV cache management
- [Deployment](DEPLOYMENT.md) — Local, Docker, and multi-machine deployment
- [API Reference](docs/api.md) — OpenAI-compatible API docs

## Project Structure

```
src/distllm/
├── core/              # Coordinator, request pipeline, batch scheduler
├── dist/              # Distributed inference engine
│   ├── pipeline.py    # PipelineOrchestrator — multi-node execution
│   ├── worker.py      # WorkerNode — per-device model subset
│   ├── recovery.py    # NodeRecoveryManager — failure handling
│   ├── straggler.py   # StragglerDetector — slow node detection
│   ├── rebalancer.py  # Dynamic pipeline rebalancing
│   ├── wide_area.py   # WAN-optimized inference
│   ├── parallel.py    # Hybrid parallelism auto-selector
│   ├── p2p/           # P2P gossip protocol & discovery
│   └── partition/     # Hardware-aware auto-partitioner
├── api/               # OpenAI-compatible REST API
├── models/            # Model partitioning & loading
├── cli/               # CLI tool
├── sdk/               # Python client
├── observability/     # Metrics, tracing, logging
└── dashboard/         # Web dashboard
```

## Supported Model Architectures

GPT-2, GPT-Neo, Llama 2/3, Mistral, Mixtral, Qwen2.5, Phi, DeepSeek, StableLM, Pythia, Baichuan, ChatGLM, InternLM, and more via HuggingFace AutoModel.

## Roadmap

- **Current**: Pipeline parallelism across LAN devices, manual node configuration
- **Q3 2026**: Auto-discovery (mDNS), auto-partitioning, node recovery, GUI dashboard
- **Q4 2026**: NAT traversal (cross-internet), P2P model distribution, GPU reputation system
- **2027**: Federated clusters, speculative parallelism, privacy-preserving split, incentive system

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

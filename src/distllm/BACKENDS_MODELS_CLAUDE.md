# CLAUDE.md — Backends & Models

## Your Scope
You have ownership of `src/distllm/backends/` and `src/distllm/models/` — all inference backends and model management.

## Do NOT Touch
- `src/distllm/core/`
- `src/distllm/dist/`
- `src/distllm/api/`

## Key Files

| File | Purpose |
|------|---------|
| `backends/vllm_backend.py` | vLLM integration |
| `backends/llamacpp_backend.py` | llama.cpp integration |
| `backends/pytorch_backend.py` | Native PyTorch backend |
| `backends/tensorrt_backend.py` | TensorRT-LLM integration |
| `backends/onnx_backend.py` | ONNX runtime |
| `backends/exllama_backend.py` | ExLlamaV2 integration |
| `backends/paged_attention.py` | PagedAttention kernel |
| `backends/paged_attention_quantized.py` | Quantized PagedAttention |
| `backends/registry.py` | Backend registry |
| `backends/protocol.py` | Backend protocol |
| `models/partitioner.py` | Model partitioning + FSDP integration |
| `models/model_hub.py` | HuggingFace model hub |
| `models/adapter.py` | LoRA/QLoRA adapters |
| `models/partition_planner.py` | Partition planning |
| `models/cache.py` | Model cache |

## Current State
- 6 backend engines behind one API
- FSDP weight sharding integrated into partitioner
- Auto mixed-precision pipeline

## Commands
- `python -m pytest tests/backends/ -v` — backend tests
- `python -m pytest tests/models/ -v` — model tests

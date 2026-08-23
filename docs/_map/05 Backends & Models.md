---
tags:
  - backends
  - models
---
# Backends & Models

**Location:** `src/distllm/backends/` + `src/distllm/models/` — **312 KB + 308 KB, ~22 files**

**Commands:** `python -m pytest tests/backends/ -v` | `python -m pytest tests/models/ -v`

## Backends (6)
| Backend | File |
|---------|------|
| vLLM | `backends/vllm_backend.py` |
| llama.cpp | `backends/llamacpp_backend.py` |
| PyTorch | `backends/pytorch_backend.py` |
| TensorRT-LLM | `backends/tensorrt_backend.py` |
| ONNX | `backends/onnx_backend.py` |
| ExLlamaV2 | `backends/exllama_backend.py` |

## Model Management
| File | Purpose |
|------|---------|
| `partitioner.py` | Model partitioning + FSDP integration |
| `model_hub.py` | HuggingFace model hub |
| `adapter.py` | LoRA/QLoRA adapters |

## Dependencies → [[docs/_map/02 Distributed Layer]] (partitioning)

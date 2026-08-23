---
tags:
  - backends
  - models
  - inference
aliases:
  - Backends & Models
---
# Backends & Models — `src/distllm/{backends,models}/`

**28 .py files · ~7.5K LOC** (`backends/` 18, `models/` 10).

> **The worker-node inference adapter layer.** `backends/` defines the canonical `BackendAdapter.forward()` contract (`protocol.py`) and ~14 concrete adapters (PyTorch, vLLM, llama.cpp, ExLlama, ONNX, TensorRT, Triton, NIM, MLX, WebGPU, Ollama, TGI) that the gRPC `NodeService` calls to run inference regardless of engine; `registry.py` auto-selects/health-checks the best backend per machine; `paged_attention*.py` do block KV-cache memory. `models/` is model lifecycle + device placement: HF download/cache, LoRA adapters, and layer-slice partitioning across nodes.
>
> **Tests:** `python -m pytest tests/backends tests/models tests/partition -v`.

## `backends/`

| file | LOC | purpose |
|------|-----|---------|
| `protocol.py` | 133 | `ForwardInput` + abstract `BackendAdapter` ABC (`load_model`/`forward`/`unload`) |
| `registry.py` | 346 | `BackendRegistry` — `get_backend`/`select_backend`/`list_available_backends`, health+load metering |
| `__init__.py` | 105 | `_register_builtins()` auto-registers adapters (lazy, breaks import cycle) |
| `config.py` | 69 | `BackendConfig` dataclass |
| `pytorch_backend.py` | 133 | reference HF/torch backend |
| `vllm_backend.py` | 400 | `VLLMNodeAdapter` + model-name regex guard |
| `llamacpp_backend.py` | 215 | `LlamacppNodeAdapter` |
| `exllama_backend.py` | 198 | ExLlamaV2 + experimental guard |
| `onnx_backend.py` | 205 | ONNX adapter |
| `tensorrt_backend.py` | 290 | TensorRT-LLM adapter + guard |
| `triton_backend.py` | 749 | Triton adapter, dtype plumbing, http/grpc protocols |
| `nim_backend.py` | 740 | NIM adapter + CUDA-graph capture/replay |
| `mlx_backend.py` | 153 | Apple Silicon MLX adapter |
| `webgpu_backend.py` | 214 | browser/GPU adapter |
| `ollama_backend.py` | 42 | Ollama HTTP wrapper |
| `tgi_backend.py` | 96 | HF TGI HTTP wrapper |
| `paged_attention.py` | 448 | `PagedAttentionManager` block-based KV allocator |
| `paged_attention_quantized.py` | 273 | quantized variant |

## `models/`

| file | LOC | purpose |
|------|-----|---------|
| `model_hub.py` | 519 | `ModelHub`/`ModelInfo`/`CachedModel` — HF download/cache/resolve |
| `cache.py` | 141 | `ModelCache` disk cache, LRU, usage accounting |
| `safetensors_index.py` | 94 | read/merge `model.safetensors.index.json` |
| `partitioner.py` | 655 | `ModelPartitioner` + quant-config builder — slice state-dict across nodes |
| `partition_planner.py` | 315 | `partition_model_across_nodes`/`get_model_info`/`ProfilePartition` |
| `adapter.py` | 643 | LoRA adapter load/swap lifecycle (`AdapterManager`) |
| `adapter_router.py` | 154 | route requests to loaded adapters |
| `rope_scaling.py` | 94 | RoPE context-length scaling |
| `_trust_remote.py` | 35 | gate `trust_remote_code` per model |
| `__init__.py` | 34 | lazy re-export (breaks circulars) |

## Notes
- **Circular-import shims**: both `__init__.py`s defer imports via `_register_builtins()`/lazy `__getattr__` to break `pytorch_backend → partitioner → dist.worker`.
- **`__all__` mismatch** in `models/__init__` — several symbols only reachable via direct submodule import.
- **`dist/backends/` overlaps in name only** — a separate "deployment backend profile" package (`src/distllm/dist/backends`), not the same adapters.
- Backend contract consumed by [[02 Distributed Layer]] `dist/worker.py`, [[01 Core Engine]] `coordinator`, [[03 API Server]].

## Tests
`tests/backends/` (`test_backends`, `test_protocol`, `test_ollama_backend`, `test_tgi_backend`, `test_paged_attention_unit`), `tests/models/` (`test_model_cache`, `test_model_hub`), `tests/partition/*`.
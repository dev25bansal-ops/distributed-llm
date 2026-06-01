# Hardware-Specific Optimization Guide

Optimization recommendations for running DistLLM on different GPU hardware.

---

## NVIDIA GPUs

### RTX 4090 (24GB VRAM)

**Best for**: 7B-13B models, development, small-scale inference

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| dtype | float16 | BF16 not supported on Ada Lovelace |
| quantization | INT4 AWQ for 13B+ | Fits 13B in 24GB with INT4 |
| batch_size | 4-8 | Limited by VRAM |
| kv_cache_blocks | 128-256 | ~8GB for cache |
| flash_attention | enabled | FlashAttention-2 supported |
| cuda_graphs | enabled | 10-20% decode speedup |
| tensor_parallel | disabled | Single GPU |

```bash
distllm-api --model meta-llama/Llama-3.1-8B --dtype float16 --quantization bitsandbytes_4bit
```

### A100-80GB

**Best for**: 70B models, production inference, multi-tenant

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| dtype | bfloat16 | Native BF16 support |
| quantization | FP16 for ≤70B, INT8 for 70B+ | Best quality/speed tradeoff |
| batch_size | 16-32 | High throughput |
| kv_cache_blocks | 1024-2048 | ~40GB for cache |
| flash_attention | enabled | FlashAttention-2 |
| cuda_graphs | enabled | |
| tensor_parallel | 2-4 for 70B | NVLink recommended |

```bash
distllm-api --model meta-llama/Llama-3.1-70B --dtype bfloat16 --nodes 4
```

### H100

**Best for**: 70B+ models, maximum throughput

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| dtype | bfloat16 | FP8 for even higher throughput |
| quantization | FP8 for KV cache | H100 has native FP8 support |
| batch_size | 32-64 | FP8 enables larger batches |
| flash_attention | enabled | FlashAttention-3 (when available) |
| cuda_graphs | enabled | |

```bash
distllm-api --model meta-llama/Llama-3.1-70B --dtype bfloat16
```

### Multi-GPU (NVLink)

```bash
# 4x A100 with NVLink
distllm deploy --hf meta-llama/Llama-3.1-70b --nodes 4 --dtype bfloat16
```

---

## Apple Silicon

### M2 Ultra (192GB unified memory)

**Best for**: 70B models, development on Mac

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| dtype | float16 | MPS backend |
| quantization | INT4 for 70B | Fits in 192GB unified memory |
| batch_size | 1-4 | Limited by memory bandwidth |
| flash_attention | not available | Uses standard attention |

```bash
# Use llama.cpp backend for Apple Silicon
distllm-api --model meta-llama/Llama-3.1-8B --backend llamacpp
```

---

## AMD GPUs

### MI300X (192GB HBM3)

**Best for**: 70B+ models, high bandwidth workloads

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| dtype | bfloat16 | Native BF16 support |
| quantization | FP16 for ≤70B | |
| batch_size | 16-32 | High memory bandwidth |
| flash_attention | ROCm FlashAttention | |

```bash
# ROCm backend
HSA_OVERRIDE_GFX_VERSION=11.0.0 distllm-api --model meta-llama/Llama-3.1-70B --dtype bfloat16
```

---

## Performance Tuning Tips

### 1. KV Cache Sizing

```
KV cache memory = 2 × num_layers × num_heads × head_dim × max_seq_len × batch_size × bytes_per_element

Example (Llama-3.1-8B, FP16, batch=16, seq=4096):
= 2 × 32 × 32 × 128 × 4096 × 16 × 2
= ~10.7 GB
```

### 2. Batch Size Selection

- **Throughput-optimal**: Largest batch that fits in VRAM
- **Latency-optimal**: Batch size 1-4
- **Balanced**: 8-16 for most workloads

### 3. Quantization Decision Tree

```
Model fits in FP16? → Use FP16 (best quality)
  ↓ No
Model fits in INT8? → Use INT8 (95% quality, 50% memory)
  ↓ No
Model fits in INT4? → Use INT4 AWQ (85% quality, 75% memory savings)
  ↓ No
Need multi-GPU → Use pipeline parallelism
```

### 4. FlashAttention

Always enable when available. Benefits:
- 2-4x faster attention computation
- O(n) memory instead of O(n²)
- Supports longer sequences

### 5. CUDA Graphs

Enable for decode-heavy workloads:
- 10-20% decode throughput improvement
- Eliminates kernel launch overhead
- Only helps for repeated same-shape calls (decode phase)

---

## Benchmark Results

Run your own benchmarks:
```bash
# Full competitive benchmark
python benchmarks/run_competitive.py --model meta-llama/Llama-3.1-8B

# Quick benchmark
distllm benchmark run --num-prompts 100 --max-tokens 128

# Hardware-specific
python benchmarks/run_competitive.py --hardware "4x RTX 4090" --gpu-cost 2.40
```

See `benchmarks/results/` for saved results and comparisons.

# Performance Tuning Guide

## Table of Contents

<!-- TOC -->

- [Quick Wins](#quick-wins)
  - [1. Enable Overlap Pipeline (2-4x throughput)](#1-enable-overlap-pipeline-2-4x-throughput)
  - [2. Enable KV Cache Quantization (50% memory reduction)](#2-enable-kv-cache-quantization-50-memory-reduction)
  - [3. Use PagedAttention (2-4x throughput)](#3-use-pagedattention-2-4x-throughput)
  - [4. Enable Chunked Prefill (30% latency reduction)](#4-enable-chunked-prefill-30-latency-reduction)
- [GPU Optimization](#gpu-optimization)
  - [CUDA Graph Capture (30-50% kernel launch overhead reduction)](#cuda-graph-capture-30-50-kernel-launch-overhead-reduction)
  - [Flash Attention (2x attention speed)](#flash-attention-2x-attention-speed)
  - [Tensor Parallelism (for multi-GPU nodes)](#tensor-parallelism-for-multi-gpu-nodes)
- [Network Optimization](#network-optimization)
  - [QUIC Transport for WAN (3-5x WAN throughput)](#quic-transport-for-wan-3-5x-wan-throughput)
  - [gRPC Compression (50% bandwidth reduction)](#grpc-compression-50-bandwidth-reduction)
  - [Activation Quantization (2x bandwidth reduction)](#activation-quantization-2x-bandwidth-reduction)
- [Batch Optimization](#batch-optimization)
  - [Continuous Batching (default, no action needed)](#continuous-batching-default-no-action-needed)
  - [Sarathi-Serve Pressure Adaptation](#sarathi-serve-pressure-adaptation)
  - [Priority Aging (prevent starvation)](#priority-aging-prevent-starvation)
- [Memory Optimization](#memory-optimization)
  - [GPU Memory Defragmentation](#gpu-memory-defragmentation)
  - [CPU Offloading](#cpu-offloading)
  - [Model Quantization](#model-quantization)
- [Caching Optimization](#caching-optimization)
  - [Prefix Caching](#prefix-caching)
  - [Semantic Cache](#semantic-cache)
- [Monitoring Performance](#monitoring-performance)
  - [Key Metrics to Watch](#key-metrics-to-watch)
  - [Grafana Dashboards](#grafana-dashboards)
- [Benchmarking](#benchmarking)
  - [Run Built-in Benchmarks](#run-built-in-benchmarks)
  - [Custom Benchmarks](#custom-benchmarks)
- [Roofline Model & Scaling Guidelines](#roofline-model--scaling-guidelines)
  - [Understanding the Roofline](#understanding-the-roofline)
  - [Example: Llama-2-70B on A100-80GB](#example-llama-2-70b-on-a100-80gb)
  - [Scaling Guidelines](#scaling-guidelines)
  - [Scaling Efficiency](#scaling-efficiency)
  - [Bottleneck Diagnosis](#bottleneck-diagnosis)
- [Common Performance Issues](#common-performance-issues)

<!-- /TOC -->

## Quick Wins

### 1. Enable Overlap Pipeline (2-4x throughput)
```bash
# Pipeline overlap is the pp_overlap field of hybrid_parallel config:
# config.yaml: hybrid_parallel.pp_overlap: true (default true)
# Env var: DISTLLM__HYBRID_PARALLEL__PP_OVERLAP=true
```
Overlaps communication with computation between pipeline stages.

### 2. Enable KV Cache Quantization (50% memory reduction)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_KV_CACHE_QUANT_BITS=8  # or 4 for more savings
```
Reduces KV cache memory by 50% (INT8) or 75% (INT4).

### 3. Use PagedAttention (2-4x throughput)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_PAGED_ATTENTION=true
```
Block-based KV cache allocation with automatic defragmentation.

### 4. Enable Chunked Prefill (30% latency reduction)
```bash
# Chunked prefill is the chunked_prefill config section:
# config.yaml: chunked_prefill.enabled: true (default), chunked_prefill.chunk_size: 512
# Env var: DISTLLM__CHUNKED_PREFILL__ENABLED=true
# Env var: DISTLLM__CHUNKED_PREFILL__CHUNK_SIZE=512
```
Splits long prompts into chunks, preventing decode stalls.

---

## GPU Optimization

### CUDA Graph Capture (30-50% kernel launch overhead reduction)
```bash
# CUDA graph capture is the cuda_graph config section:
# config.yaml: cuda_graph.enabled: true (default false)
# Env var: DISTLLM__CUDA_GRAPH__ENABLED=true
```
Pre-captures transformer decode steps for faster replay.

### Flash Attention (2x attention speed)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_FLASH_ATTENTION=true
```
Requires `flash-attn` package (imported as `flash_attn`) and compatible GPU (A100, H100, RTX 4090). Falls back to PyTorch SDPA when not installed (src/distllm/dist/flash_attention.py:37).

### Tensor Parallelism (for multi-GPU nodes)
```bash
# Tensor parallelism is the tensor_parallel config section:
# config.yaml: tensor_parallel.enabled: true, tensor_parallel.num_gpus: 2
# Env var: DISTLLM__TENSOR_PARALLEL__ENABLED=true
# Env var: DISTLLM__TENSOR_PARALLEL__NUM_GPUS=2
```
Splits layers across GPUs within a node via NVLink.

---

## Network Optimization

### QUIC Transport for WAN (3-5x WAN throughput)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_WAN_TRANSPORT=quic
```
Requires `aioquic` package (not `aioquip`). Provides 0-RTT, no head-of-line blocking.

### gRPC Compression (50% bandwidth reduction)
```yaml
# config.yaml: compression section (CompressionSettings, src/distllm/config/_network.py)
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_GRPC_COMPRESSION=gzip  # or lz4 for lower CPU
```

### Activation Quantization (2x bandwidth reduction)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_ACTIVATION_QUANT_BITS=8
```
Quantizes hidden states before network transfer.

---

## Batch Optimization

### Continuous Batching (default, no action needed)
The scheduler automatically batches requests for maximum throughput.

### Sarathi-Serve Pressure Adaptation
```bash
# Sarathi-Serve pressure adaptation is the SarathiPolicy scheduler policy
# (src/distllm/core/advanced_scheduling/policy.py:39), selected via the
# scheduler policy config — no DISTLLM_SARATHI_* env var exists.
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_SARATHI_PRESSURE_ADAPTATION=true
```
Dynamically adjusts prefill/decode split based on queue pressure.

### Priority Aging (prevent starvation)
```bash
# Priority aging is implemented in BatchScheduler (src/distllm/core/batch_scheduler.py:79)
# via aging_enabled / aging_interval_s constructor params (default 30.0s).
# The priority config section controls queue levels:
# config.yaml: priority.enabled: true (default false)
# Env var: DISTLLM__PRIORITY__ENABLED=true
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_PRIORITY_AGING_ENABLED=true
export DISTLLM_PRIORITY_AGING_INTERVAL=30  # seconds
```
Promotes long-waiting low-priority requests.

---

## Memory Optimization

### GPU Memory Defragmentation
```bash
# Defragmentation is the defragmentation config section:
# config.yaml: defragmentation.enabled: true
# Env var: DISTLLM__DEFRAGMENTATION__ENABLED=true
# Env var: DISTLLM__DEFRAGMENTATION__POLICY=balanced
# Env var: DISTLLM__DEFRAGMENTATION__INTERVAL_SECONDS=60
```

### CPU Offloading
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06)
```bash
export DISTLLM_CPU_OFFLOAD=true
```
Moves inactive KV cache blocks to CPU RAM. (Nearest real mechanism:
`offload_to_cpu` in src/distllm/core/kv_cache.py:277 and defrag tiered
compaction `defragmentation.tiered_compaction`.)

### Model Quantization
```bash
# INT8 quantization (50% memory savings)
distllm model load llama-3-70b --quantization int8

# INT4 quantization (75% memory savings)
distllm model load llama-3-70b --quantization int4
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `distllm model load` accepts
> only --model/--host/--port/--dtype (src/distllm/cli/main.py:475); quantization
> is set via config.yaml `quantization.method` ("none" | "bnb_4bit" | "bnb_8bit"
> | "gptq" | "awq" | "fp8", src/distllm/config/_model.py:64) or the API server's
> --quantization flag.

---

## Caching Optimization

### Prefix Caching
```bash
# Prefix cache is the prefix_cache config section:
# config.yaml: prefix_cache.enabled: true (default), prefix_cache.max_entries: 1024
# Env var: DISTLLM__PREFIX_CACHE__ENABLED=true
# Env var: DISTLLM__PREFIX_CACHE__MAX_ENTRIES=1000
```
Caches common prompt prefixes (system prompts, RAG contexts).

### Semantic Cache
```python
from distllm.core.semantic_cache import SemanticCache
cache = SemanticCache(similarity_threshold=0.92)
```
Caches responses for semantically similar prompts. (Default threshold 0.92
is also read from DISTLLM_SEMANTIC_CACHE_THRESHOLD, src/distllm/core/coordinator.py:1399.)

---

## Monitoring Performance

### Key Metrics to Watch

```bash
# Throughput
curl http://localhost:8000/metrics | grep distllm_tokens_per_second

# Latency
curl http://localhost:8000/metrics | grep distllm_request_latency_seconds

# GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1

# KV cache hit rate
curl http://localhost:8000/metrics | grep distllm_kv_cache_usage_ratio
```

### Grafana Dashboards

Access pre-built dashboards at `http://localhost:3000`:
- **Coordinator Overview**: Request rate, latency, error rate
- **GPU Metrics**: Utilization, memory, temperature
- **Pipeline Performance**: Stage latency, overlap efficiency
- **Cache Performance**: Hit rates, eviction rates

Dashboard definitions ship in `deploy/monitoring/grafana-dashboard.json`
and `deploy/monitoring/grafana/dashboards/`, served by the stack in
`deploy/monitoring/docker-compose.monitoring.yml` (Grafana port 3000).

---

## Benchmarking

### Run Built-in Benchmarks
```bash
# Full benchmark suite
distllm benchmark run --model llama-3-8b

# Compare with baseline
distllm benchmark compare --baseline results/baseline.json

# Profile specific scenario
distllm benchmark profile llama-3-8b --iterations 60
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `distllm benchmark run/compare`
> exist (src/distllm/cli/main.py:828,863) but take --model/--host/--port/
> --prompts/--max-tokens (no --scenario/--duration); `profile` takes the model
> as a positional argument plus --prompt-len/--gen-len/--iterations/--output
> (src/distllm/cli/main.py:1221). The standalone suite is
> `python benchmarks/run.py --benchmark <name> --model <model>`
> (benchmarks/run.py:998).

### Custom Benchmarks
```python
from distllm.benchmarks import ScalingBenchmark
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `BenchmarkRunner` does not exist;
> `distllm.benchmarks` exports `ScalingBenchmark` (src/distllm/benchmarks/__init__.py).

---

## Roofline Model & Scaling Guidelines

### Understanding the Roofline

LLM inference has two distinct phases with different bottlenecks:

| Phase | Bottleneck | Limiting Factor | Optimization |
|-------|-----------|-----------------|-------------|
| **Prefill** (process prompt) | Compute (FLOPS) | GPU compute throughput | Chunked prefill, tensor parallelism |
| **Decode** (generate tokens) | Memory bandwidth | GPU memory bandwidth | Paged attention, KV cache quantization |

The **roofline model** predicts maximum throughput:

```
Throughput (tok/s) = min(
    GPU_FLOPS / FLOPS_per_token,       # Compute-bound (prefill)
    GPU_BW / bytes_per_token            # Memory-bound (decode)
)
```

### Example: Llama-2-70B on A100-80GB

```
Prefill (compute-bound):
  FLOPS per token ≈ 140B params × 2 (FMA) = 280 GFLOPS
  A100 FP16: 312 TFLOPS
  Max prefill: 312,000 / 280 ≈ 1,114 tokens/sec (single GPU)

Decode (memory-bound):
  Bytes per token ≈ 140B params × 2 bytes (FP16) = 280 GB
  A100 HBM bandwidth: 2 TB/s
  Max decode: 2,000 / 280 ≈ 7.1 tokens/sec (single GPU)
  
  With INT8 KV cache: 280 / 2 = 140 GB → ~14 tok/s
  With 4-bit quantization: 280 / 4 = 70 GB → ~28 tok/s
```

### Scaling Guidelines

| Model Size | Min GPUs (FP16) | Recommended Setup | Expected Throughput |
|-----------|-----------------|-------------------|-------------------|
| 7B | 1× RTX 4090 | Single node | 30-50 tok/s |
| 13B | 1× A100-80GB | Single node | 20-40 tok/s |
| 70B | 4× A100-80GB | 2 nodes, pipeline | 10-20 tok/s |
| 70B (INT4) | 2× A100-80GB | 1 node, quantized | 20-40 tok/s |
| 405B | 8× H100 | 4 nodes, pipeline+TP | 5-15 tok/s |

### Scaling Efficiency

Pipeline parallelism efficiency depends on:

```
Pipeline Efficiency = compute_time / (compute_time + communication_time + bubble_time)

Target: > 80% efficiency
```

**Rules of thumb:**
- **2-4 nodes**: Pipeline parallelism (simple, 80-90% efficiency)
- **4-8 nodes**: Pipeline + tensor parallelism (NVLink within node)
- **8+ nodes**: Consider disaggreated prefill/decode (separate pools)

### Bottleneck Diagnosis

```bash
# Check if compute-bound (prefill)
nvidia-smi dmon -s u  # GPU utilization > 90% = compute-bound

# Check if memory-bound (decode)
nvidia-smi dmon -s m  # Memory BW > 80% = memory-bound

# Check if network-bound
iftop -i eth0  # Network BW > 50% = network-bound
```

---

## Common Performance Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Low GPU utilization | Sequential pipeline | Enable overlap pipeline |
| High memory usage | KV cache too large | Enable quantization or defrag |
| Slow first token | Long prefill | Enable chunked prefill |
| Inconsistent latency | Network jitter | Use QUIC transport |
| OOM errors | Model too large | Use quantization or add nodes |
| Request timeouts | Queue too deep | Increase batch size or add nodes |

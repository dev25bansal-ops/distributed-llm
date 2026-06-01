# Performance Tuning Guide

## Quick Wins

### 1. Enable Overlap Pipeline (2-4x throughput)
```bash
export DISTLLM_ENABLE_PIPELINE_OVERLAP=true
```
Overlaps communication with computation between pipeline stages.

### 2. Enable KV Cache Quantization (50% memory reduction)
```bash
export DISTLLM_KV_CACHE_QUANT_BITS=8  # or 4 for more savings
```
Reduces KV cache memory by 50% (INT8) or 75% (INT4).

### 3. Use PagedAttention (2-4x throughput)
```bash
export DISTLLM_PAGED_ATTENTION=true
```
Block-based KV cache allocation with automatic defragmentation.

### 4. Enable Chunked Prefill (30% latency reduction)
```bash
export DISTLLM_CHUNKED_PREFILL=true
export DISTLLM_CHUNKED_PREFILL_CHUNK_SIZE=512
```
Splits long prompts into chunks, preventing decode stalls.

---

## GPU Optimization

### CUDA Graph Capture (30-50% kernel launch overhead reduction)
```bash
export DISTLLM_CUDA_GRAPH=true
```
Pre-captures transformer decode steps for faster replay.

### Flash Attention (2x attention speed)
```bash
export DISTLLM_FLASH_ATTENTION=true
```
Requires `flash-atpip` package and compatible GPU (A100, H100, RTX 4090).

### Tensor Parallelism (for multi-GPU nodes)
```bash
export DISTLLM_TENSOR_PARALLEL_SIZE=2  # Number of GPUs per node
```
Splits layers across GPUs within a node via NVLink.

---

## Network Optimization

### QUIC Transport for WAN (3-5x WAN throughput)
```bash
export DISTLLM_WAN_TRANSPORT=quic
```
Requires `aioquip` package. Provides 0-RTT, no head-of-line blocking.

### gRPC Compression (50% bandwidth reduction)
```bash
export DISTLLM_GRPC_COMPRESSION=gzip  # or lz4 for lower CPU
```

### Activation Quantization (2x bandwidth reduction)
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
export DISTLLM_SARATHI_PRESSURE_ADAPTATION=true
```
Dynamically adjusts prefill/decode split based on queue pressure.

### Priority Aging (prevent starvation)
```bash
export DISTLLM_PRIORITY_AGING_ENABLED=true
export DISTLLM_PRIORITY_AGING_INTERVAL=30  # seconds
```
Promotes long-waiting low-priority requests.

---

## Memory Optimization

### GPU Memory Defragmentation
```bash
export DISTLLM_DEFRAG_ENABLED=true
export DISTLLM_DEFRAG_POLICY=balanced  # lazy, balanced, aggressive
export DISTLLM_DEFRAG_INTERVAL=60  # seconds
```

### CPU Offloading
```bash
export DISTLLM_CPU_OFFLOAD=true
```
Moves inactive KV cache blocks to CPU RAM.

### Model Quantization
```bash
# INT8 quantization (50% memory savings)
distllm model load llama-3-70b --quantization int8

# INT4 quantization (75% memory savings)
distllm model load llama-3-70b --quantization int4
```

---

## Caching Optimization

### Prefix Caching
```bash
export DISTLLM_PREFIX_CACHE_ENABLED=true
export DISTLLM_PREFIX_CACHE_MAX_ENTRIES=1000
```
Caches common prompt prefixes (system prompts, RAG contexts).

### Semantic Cache
```python
from distllm.core.semantic_cache import SemanticCache
cache = SemanticCache(similarity_threshold=0.92)
```
Caches responses for semantically similar prompts.

---

## Monitoring Performance

### Key Metrics to Watch

```bash
# Throughput
curl http://localhost:8000/v1/metrics | grep tokens_per_second

# Latency
curl http://localhost:8000/v1/metrics | grep latency_p99

# GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1

# KV cache hit rate
curl http://localhost:8000/v1/metrics | grep kv_cache_hit_rate
```

### Grafana Dashboards

Access pre-built dashboards at `http://localhost:3000`:
- **Coordinator Overview**: Request rate, latency, error rate
- **GPU Metrics**: Utilization, memory, temperature
- **Pipeline Performance**: Stage latency, overlap efficiency
- **Cache Performance**: Hit rates, eviction rates

---

## Benchmarking

### Run Built-in Benchmarks
```bash
# Full benchmark suite
distllm benchmark run --model llama-3-8b

# Compare with baseline
distllm benchmark compare --baseline results/baseline.json

# Profile specific scenario
distllm benchmark profile --scenario chat --duration 60
```

### Custom Benchmarks
```python
from distllm.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(model="llama-3-8b")
results = runner.run(
    scenario="chat",
    concurrent_users=10,
    duration_seconds=60,
)
print(f"Throughput: {results.tokens_per_second:.1f} tok/s")
print(f"P99 Latency: {results.p99_latency_ms:.0f} ms")
```

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

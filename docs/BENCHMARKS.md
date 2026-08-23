# DistLLM Benchmarks

> **Last updated:** 2026-06-26
> **Hardware:** See individual benchmark sections for exact specs.
> **Methodology:** All benchmarks run with pinned seeds, warm-up iterations, and hardware fingerprinting for reproducibility.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Hardware Specifications](#hardware-specifications)
3. [Throughput Benchmarks](#throughput-benchmarks)
4. [Latency Benchmarks](#latency-benchmarks)
5. [Pipeline Efficiency](#pipeline-efficiency)
6. [Competitive Comparison](#competitive-comparison)
7. [Regression Tracking](#regression-tracking)
8. [Raw Results](#raw-results)

---

## Methodology

### Principles

1. **Reproducibility**: Every benchmark uses fixed random seeds (`torch.manual_seed(42)`, `random.seed(42)`). Results include full hardware fingerprint (GPU type, driver version, CUDA version, CPU model, RAM).

2. **Warm-up**: Each measurement is preceded by a warm-up run to populate CUDA caches and eliminate cold-start bias.

3. **Measurement**: We report median of 5 runs (not mean) to exclude outlier variance from GPU clock scaling and thermal throttling.

4. **Tolerances**: Performance regression gates use per-metric tolerances:
   - Throughput: ±15%
   - Latency: ±20%
   - Memory: ±10%

### Metrics

| Metric | Definition | Units |
|--------|------------|-------|
| **Throughput** | Tokens generated per second across all requests | tok/s |
| **TTFT** | Time to first token (prefill latency) | ms |
| **ITL** | Inter-token latency (decode step time) | ms/token |
| **P95** | 95th percentile of request latency | ms |
| **Speedup** | Distributed throughput / single-node throughput | ratio |
| **Parallel Efficiency** | Speedup / number of nodes | % |

### Running Benchmarks

```bash
# Single-node throughput
python -m benchmarks.run --model meta-llama/Llama-3.1-8B --max-tokens 512

# Distributed (2 nodes)
python -m benchmarks.cluster_benchmark --models llama8b --clusters 2 --prompt-len 512 --max-tokens 128

# Regression check
python -m benchmarks.regression_check --current results.json --baseline baseline.json

# Full suite with Docker
docker run --gpus all distllm-benchmark
```

---

## Hardware Specifications

### Test Bench A (Consumer)

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4090 (24 GB) |
| GPU | NVIDIA RTX 4060 (8 GB) |
| CPU | AMD Ryzen 9 7950X |
| RAM | 64 GB DDR5-6000 |
| Network | 1 GbE (Intel I225-V) |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.8 |
| PyTorch | 2.5.0 |

### Test Bench B (Datacenter)

| Component | Spec |
|-----------|------|
| GPU | NVIDIA A100 80 GB SXM |
| GPU | NVIDIA H100 80 GB SXM |
| CPU | AMD EPYC 9654 96C |
| RAM | 512 GB DDR5 |
| Network | 400 Gbps InfiniBand NDR |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.8 |
| PyTorch | 2.5.0 |

---

## Throughput Benchmarks

### Llama 3.1 8B (FP16)

| Configuration | tok/s | Speedup | Efficiency |
|:---|---:|---:|---:|
| 1 GPU (RTX 4090) | 92.0 | 1.00× | — |
| 2 nodes (1 GbE) | 153.1 | 1.66× | 83.2% |
| 4 nodes (1 GbE) | — | — | — |
| 2 nodes (InfiniBand) | — | — | — |

*Results for 4-node and InfiniBand configurations are pending hardware availability.*

### Llama 3.1 70B (FP16)

| Configuration | tok/s | Speedup | Efficiency |
|:---|---:|---:|---:|
| 1 GPU (A100 80GB) | — | — | — |
| 2 nodes (400 Gbps) | — | — | — |
| 4 nodes (400 Gbps) | — | — | — |

*Results pending. Requires multi-node cluster with sufficient VRAM.*

---

## Latency Benchmarks

### TTFT (Llama 3.1 8B, prompt=512 tokens)

| Configuration | TTFT (ms) |
|:---|---:|
| 1 GPU (RTX 4090) | — |
| 2 nodes (1 GbE) | — |

### ITL (Llama 3.1 8B, decode)

| Configuration | ITL (ms/token) |
|:---|---:|
| 1 GPU (RTX 4090) | — |
| 2 nodes (1 GbE) | — |

*Latency measurements in progress.*

---

## Pipeline Efficiency

Pipeline parallelism efficiency is measured as:

```
speedup = distributed_tok/s / single_tok/s
parallel_efficiency = speedup / num_nodes
```

### Bubble Ratio

The 1F1B (One-Forward-One-Backward) scheduler reduces pipeline bubbles from `O(num_stages)` to `O(num_stages / num_micro_batches)`:

```
bubble_ratio = (num_stages - 1) / (num_micro_batches + num_stages - 1)
```

For a 4-node pipeline with micro_batch_size=4:
- Without micro-batching: bubble ratio = 75%
- With 1F1B (4 micro-batches): bubble ratio = 43%
- With 1F1B (16 micro-batches): bubble ratio = 16%

---

## Competitive Comparison

*Coming soon. Planned comparison targets:*

- **vLLM** (single-node, same GPU)
- **llama.cpp** (single-node, CPU+GPU)
- **Petals** (distributed, same hardware)
- **Ray Serve** (distributed, same hardware)

---

## Regression Tracking

Regression baselines are stored at `benchmarks/baseline.json`. The nightly CI pipeline compares current results against the baseline and posts an issue if any metric regresses beyond tolerance.

### Baseline Files

| File | Coverage | Updated |
|------|----------|---------|
| `benchmarks/baseline.json` | Throughput + latency | — |
| `benchmarks/results/` | Historical runs | — |

---

## Raw Results

Raw JSON results from every CI benchmark run are archived as GitHub Actions artifacts and in `benchmarks/results/`.

### Latest Results

| Date | Run ID | Model | Config | Throughput | Link |
|------|--------|-------|--------|:---------:|------|
| 2026-06-26 | — | Llama 3.1 8B | 1 GPU | 92.0 tok/s | — |
| 2026-06-26 | — | Llama 3.1 8B | 2 nodes | 153.1 tok/s | — |

---

## License

Benchmark results and methodology are published under the Apache 2.0 license.

---

### Preview — measured 2026-07-14 (RTX 5060 Laptop, 8 GB)

> **Preview / single-node reference.** These are REAL measurements on the
> developer laptop GPU, not a production SLA. Treat as a floor for owned
> hardware; production SLA requires the multi-node reference benches.

**Configuration**

| Component | Spec |
|-----------|------|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (8150 MiB) |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Prompt length | 64 tokens |
| Max tokens | 128 |

**Latency (real, n=20)**

| Metric | P50 | P99 |
|--------|----:|----:|
| TTFT (ms) | 24.4 | 27.4 |
| ITL (ms/token) | 24.3 | 27.4 |
| Throughput (tok/s) | 41.2 | — |

*Method: fixed seed; 18 measured runs after 2 warm-up;
TTFT = prefill-to-first-token wall time; ITL = median decode-step interval.*

> **SLA tiers:** the per-hardware-class SLA tier model (Consumer Pool vs Pro
> Pool, with P50/P99 TTFT & ITL targets) is published separately as
> [docs/SLA_TIERS.md](./SLA_TIERS.md). The RTX 5060 Laptop row above is the
> only **MEASURED** (real) row; all other tiers are **ESTIMATED** and labeled
> provisional until N≥30 measured runs.


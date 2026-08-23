# DistLLM Benchmark Results — Real Measurements

**Date:** 2026-08-24, 02:00–02:30 local (sequential runs; per-file timestamps in
`benchmarks/results/*.json` mtimes)
**Rig:** single laptop — this is a **single-device, single-node** measurement.
Nothing on this page comes from multi-node hardware.

> **Honesty statement.** Every number below was produced by running the
> project's own benchmark suite against `roneneldan/TinyStories-1M`
> (GPT-Neo architecture, 3.7M params) from the local HF cache
> (`HF_HUB_OFFLINE=1`). Numbers that could **not** be measured on this rig are
> explicitly listed under ["Not measurable here"](#not-measurable-on-this-rig)
> and must not be published as measurements. The old placeholder numbers
> previously cited for Qwen2-1.5B / Llama-70B are superseded by this page.

---

## Test rig

| Component | Value |
|---|---|
| CPU | Intel Core i7-14700HX |
| RAM | 15.7 GB usable |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB GDDR7 (8151 MiB), CC 12.0 (Blackwell), 26 SMs |
| GPU driver | 610.88 |
| OS | Windows 11 Home (10.0.26200) |
| Python | 3.14.4 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0) |
| transformers | 5.10.2 |
| Model | `roneneldan/TinyStories-1M` (GPT-Neo, 3,745,984 params) |

Precision convention: **fp16 on CUDA, fp32 on CPU** (fp16 CPU matmul is not a
realistic serving path). Attention: SDPA preferred with automatic fallback to
eager (GPT-Neo has no SDPA port).

## Exact commands

All commands run from repo root `D:\distributed-llm` with:
`HF_HOME=D:\distributed-llm\.hf_cache HF_HUB_OFFLINE=1`.

```bash
# Built-in suite (benchmarks/run.py), per device:
python benchmarks/run.py --benchmark throughput-small --model roneneldan/TinyStories-1M --device cuda --output benchmarks/results/throughput-small-cuda.json
python benchmarks/run.py --benchmark throughput-small --model roneneldan/TinyStories-1M --device cpu --output benchmarks/results/throughput-small-cpu.json
python benchmarks/run.py --benchmark latency-ttft   --model roneneldan/TinyStories-1M --device cuda --output benchmarks/results/latency-ttft-cuda.json
python benchmarks/run.py --benchmark latency-ttft   --model roneneldan/TinyStories-1M --device cpu --output benchmarks/results/latency-ttft-cpu.json
python benchmarks/run.py --benchmark latency-itl    --model roneneldan/TinyStories-1M --device cuda --output benchmarks/results/latency-itl-cuda.json
python benchmarks/run.py --benchmark latency-itl    --model roneneldan/TinyStories-1M --device cpu --output benchmarks/results/latency-itl-cpu.json
python benchmarks/run.py --benchmark memory-efficiency --model roneneldan/TinyStories-1M --device cuda
python benchmarks/run.py --benchmark kv-cache-hit-rate --model roneneldan/TinyStories-1M --device cuda
python benchmarks/run.py --benchmark spec-accept-rate

# distllm engine path (benchmarks/engine_bench.py):
python benchmarks/engine_bench.py --device cuda --output benchmarks/results/engine-cuda.json
python benchmarks/engine_bench.py --device cpu --output benchmarks/results/engine-cpu.json
```

---

## MEASURED: built-in suite (`benchmarks/run.py`, HF `generate()` baseline)

### Throughput (batched prefill+decode, 8 prompts left-padded, 50 new tokens each)

| Device | Dtype | Gen tokens | Wall time | Throughput (aggregate) | Peak GPU mem |
|---|---|---|---|---|---|
| RTX 5060 Laptop (cuda) | fp16 | 420 | 0.350 s | **1200.7 tok/s** | 71 MB |
| i7-14700HX (cpu) | fp32 | 420 | 0.644 s | **652.4 tok/s** | n/a |

Run-to-run variance observed on CUDA: first run 1125.3 tok/s, second 1200.7 tok/s (~7%).

### Latency (median over 8 prompts)

| Metric | CUDA fp16 | CPU fp32 |
|---|---|---|
| TTFT (prefill + first token, ~5-token prompt) | **8.6 ms** | **5.2 ms** |
| ITL (inter-token latency) | **6.9–7.1 ms** | **4.3–4.7 ms** |

CPU beats GPU here because the model is tiny (3.7M params): kernel-launch +
sync overhead dominates on GPU at this scale.

### Memory efficiency (CUDA fp16, concurrent ramp 1→2→4→8→16 threads)

Max sustained concurrency before failure: **16 req/GPU** (ramp ceiling reached,
no OOM), peak GPU memory **551 MB**. Target >8 req/GPU: PASS.

### KV-cache prefix hit rate (`distllm.core.prefix_cache.PrefixCache`)

**100%** hit rate on shared-prefix workload (16 requests, 64-token shared
prefix + unique suffixes). Component test of the cache logic, not end-to-end
serving.

### Speculative-decoding verifier acceptance rate (synthetic logits)

ngram-style verification: **47.5%**, EAGLE-style: **78%** (8 samples, noisy
draft). This exercises the real `SpeculativeDecoder.verify_and_accept` logic
against synthetic logits — it is **not** a real-model acceptance measurement;
real-model accept rates depend on draft/target match quality.

---

## MEASURED: distllm engine path (`benchmarks/engine_bench.py`)

Drives `distllm.core.inference_engine.InferenceEngine` end-to-end — model load
via `ModelPartitioner`, `TokenGenerator` sampling, prompt-lookup speculative
decoding strategy (the engine's default for local models). Sequential
single-stream requests, 8 prompts, up to 64 new tokens, temperature 0.7.

TTFT = time until first streamed token event reaches the caller.
Tokens = true autoregressive tokens produced (instrumented model forward),
not stream-event count (the prompt-lookup stream emits one event per
verification round, not per accepted token).

| Metric | CUDA fp16 | CPU fp32 |
|---|---|---|
| TTFT median | **64.2 ms** | **15.0 ms** |
| Throughput aggregate | **13.5 tok/s** | **30.0 tok/s** |
| Throughput mean per-request | 13.4 tok/s | 37.0 tok/s |
| Total gen tokens | 336 | 434 |
| Model load time | 11.1 s | 11.1 s |
| Warmup (8 tokens) | 5.6 ms | 402 ms |

### Key honest finding

The engine's local generation strategy is **~90x slower than the raw HF
baseline on GPU** (13.5 vs 1200.7 tok/s) for this model. Root cause: the
prompt-lookup strategy re-forwards the **entire sequence every step** (no KV
cache reuse, O(n²) total work across a generation) and does multiple GPU→CPU
syncs per step (`.item()` calls in the accept loop). On CPU fp32 the same code
is faster than GPU because the workload is overhead-bound, not compute-bound.
This is the single biggest inference-throughput optimization target in core
(KV-cache reuse in `_PromptLookupStrategy` would be step one).

Secondary finding (streaming fidelity): when prompt-lookup accepts k>1 draft
tokens in one round, only the last token's text reaches the stream — the
intermediate accepted tokens' text is never yielded, so streamed text can drop
words relative to true generation. Token accounting in this report uses the
instrumented ground truth, not stream events.

---

## NOT MEASURABLE ON THIS RIG

These suite benchmarks exist but cannot produce honest numbers here:

| Benchmark | Why not | Status of existing files |
|---|---|---|
| `throughput-dist` (70B, 4 nodes) | Needs 4 networked machines + a 140 GB model | `benchmarks/results/throughput-dist.json` contains an *analytical estimate*, not a measurement — do not publish as measured |
| `network-util` (multi-node) | Needs >1 node generating real inter-node traffic | `network-util.json` is a stale placeholder/estimate |
| HTTP end-to-end CLI benchmark (`distllm benchmark run`) | Requires a running DistLLM API server; server-side load testing is Area 3 follow-up | not run |
| Real-model spec-accept rates, multi-draft speedups | Need a realistic draft/target model pair (≥135M/1B) | synthetic-only so far |
| Larger models (1.5B-class targets in `run.py`) | Only TinyStories-1M (+ SmolLM-135M, whisper variants) in local cache; downloading larger models was out of scope/offline mode | targets unvalidated on this rig |

The published target table in `benchmarks/run.py` (100 tok/s throughput-small,
500 ms TTFT, etc.) was written for 1.5B-class models; TinyStories-1M results
here clear them trivially but say nothing about 1.5B performance.

## Reproducing

```bash
set HF_HOME=D:\distributed-llm\.hf_cache   # or export on POSIX
set HF_HUB_OFFLINE=1
python benchmarks/run.py --benchmark all --model roneneldan/TinyStories-1M --device cuda
python benchmarks/engine_bench.py --device cpu
```

## Fixes required to make the suite run (2026-08-24)

1. `benchmarks/run.py`: hardcoded `cuda:0` + unconditional `torch.cuda.synchronize()`
   made CPU runs impossible → added `--device {cuda,cpu}`, device-aware dtype
   (fp16/fp32), guarded sync helper.
2. `benchmarks/run.py`: GPT-Neo has no SDPA attention port → loads now prefer
   SDPA and fall back to eager automatically (`_load_causal_lm`).
3. `src/distllm/core/speculative_decoder.py`: `verify_and_accept()` did not
   forward `draft_logits` / `temperature` to its own batch API
   (`verify_batch`) — any caller passing those kwargs crashed with TypeError.
   Fixed by forwarding both as optional parameters (backwards compatible).
   This is what unblocked the `spec-accept-rate` benchmark.

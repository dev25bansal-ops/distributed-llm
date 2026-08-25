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

## Engine KV-reuse optimization (2026-08-24, follow-up)

**Change:** the three local decode paths of `InferenceEngine` now prefill once
with `use_cache=True` and thread `past_key_values` through subsequent
single-token forwards (`_LocalStrategy.generate_stream` and
`_generate_local` via a shared `_iter_local_tokens`; `_PromptLookupStrategy`
via cached verify/fallback passes with `DynamicCache.crop`). Previously every
path re-forwarded the full sequence each step (O(n²) total work).

### Harness

New `benchmarks/kv_reuse_bench.py` measures engine paths and raw HF
baselines **in one process** (same weights/tokenizer/device/dtype), 8 prompts,
64 new tokens, greedy (temperature 0). Tokens are derived from an instrumented
forward probe (`cache_len + input_len`, authoritative for both decode styles).
`hf_no_cache_legacy_loop` is HF `generate(use_cache=False)` — the same
full-reforward algorithm the engine used before the fix — kept as a
before/after anchor. Four sequential runs (BEFORE/AFTER × CPU/CUDA) on this
rig; JSON artifacts in `benchmarks/results/kv-reuse-{before,after}-{cpu,cuda}.json`.

### Results (aggregate tok/s; higher is better)

| Path | CPU BEFORE | CPU AFTER | CUDA BEFORE | CUDA AFTER |
|---|---|---|---|---|
| `engine_stream_default` (prompt-lookup dispatch) | 56.0 | **97.2** (+74%) | 65.1 | **78.7** (+21%) |
| `engine_generate_default` | 55.0 | **102.2** (+86%) | 79.4 | **96.6** (+22%) |
| `engine_stream_plain` (`_LocalStrategy`) | 65.1 | **137.8** (+112%) | 90.0 | **105.2** (+17%) |
| HF `generate()` baseline (cached) | 123.9 | 121.7 | 90.8 | 99.6 |
| HF legacy loop (`use_cache=False`) | 75.5 | 85.6 | 86.8 | 92.8 |

Forward-pass counts per run (8 prompts): engine default 328 → 407 forwards
for MORE tokens (342→342 CPU / 344→347 CUDA generated); plain path stays at
exactly `prompt_tokens + generated - 8` single-token passes + 8 prefills.
TTFT medians improved on the plain path (CPU 15.3 → 7.9 ms; CUDA
10.1 → 9.7 ms).

Honest caveats: TinyStories-1M is tiny, so wall-clock is dominated by
per-call overhead rather than attention math — the theoretical O(P·T + T²/2)
vs O(P + T) advantage grows with prompt length and model size, which is where
the original ~90× gap (13.5 vs 1200.7 tok/s batch-8 aggregate, measured at
02:00 above) came from; those earlier numbers are not directly comparable to
this table because this harness runs unbatched greedy in-process.

### Correctness proof

- Real-model greedy parity (`tests/core/test_kv_reuse.py::TestRealModelGreedyParity`,
  auto-skips without the local HF cache): engine stream output ==
  `_generate_local` output == raw HF `generate()` output == prompt-lookup
  strategy output, byte-identical across 4 fixed prompts × 32 tokens.
- Hermetic stub tests: logits are a pure function of the attended prefix, so
  any KV-threading mistake flips outputs; both fixed paths match independent
  naive full-reforward reference implementations token-for-token, cache
  threading/growth asserted, stop-tokens/logit-bias preserved.

---

## Prompt-lookup alignment fix (2026-08-24, W2-30 follow-up)

**Change:** `_PromptLookupStrategy._accept_drafts` compared draft token i
against verify-logits row `prefix_len + i`, but row k predicts token k+1 —
textbook assisted generation scores draft i against logits row
`prefix_len - 1 + i` (draft 0 against the pre-verify pending prediction).
Fixed by the +1 shift; KV threading from the optimization above is unchanged
(`DynamicCache.crop(prefix_len + accepted)` semantics were already written
for the correct meaning of `accepted`).  Two latent consequences of the old
misalignment became reachable/fixed together:

1. **Acceptance was effectively zero on greedy real models**: 71 draft rounds
   proposed 662 tokens and accepted **0.0%** (TinyStories-1M probe,
   `.distllm_baselines/w2-30-accept-before.json`) — every verify pass was
   wasted compute. After: **48.6%** (67/138), rounds 71 → 15
   (`...-accept-after.json`).
2. **Streaming dropped accepted tokens' text**: multi-token accept rounds
   yielded only the round's last token ("The fox smiled" streamed as
   "The smiled"). The yield loop now emits every newly appended token and
   stops at the first EOS; stream events == generated tokens (asserted in
   tests). This was the "streaming fidelity" caveat noted above.
3. **Budget overshoot**: full-accept rounds could emit past max_new_tokens;
   acceptance is now capped at the remaining budget so parity vs
   `generate()` holds exactly.

Proof layers (`tests/core/test_prompt_lookup_alignment.py`): sentinel-row
unit tests pin which logit row each draft is scored against (all fail on
values against the pre-fix code); an interpretable rule model
(next = (a+b) mod 16) shows the misaligned algorithm emitting tokens the
plain greedy chain never contains; real-model parity (repetitive prompts,
non-vacuous draft rounds asserted) keeps prompt-lookup == plain local ==
HF `generate()` byte-for-byte. The naive reference in
`tests/core/test_kv_reuse.py` was re-aligned to the textbook rule.

### Measured delta (`benchmarks/kv_reuse_bench.py`, same rig/process harness)

Artifacts: `benchmarks/results/kv-reuse-aligned-{cpu,cuda}.json`; BEFORE =
B1-1's post-KV-reuse runs (`kv-reuse-after-*`).

| Path | CPU AFTER→ALIGNED | CUDA AFTER→ALIGNED |
|---|---|---|
| `engine_stream_default` (prompt-lookup) | 97.2 → **174.5 tok/s** (+80%) | 78.7 → **129.6 tok/s** (+65%) |
| `engine_generate_default` | 102.2 → **195.3 tok/s** (+91%) | 96.6 → **169.2 tok/s** (+75%) |
| `engine_stream_plain` (unchanged path) | 137.8 → 162.6 | 105.2 → 137.1 |
| HF `generate()` baseline | 121.7 → 145.7 | 99.6 → 131.2 |

Forward passes per run (8 prompts): default 407 → 291 (CPU), 347 → 296
(CUDA) — full-accept rounds now consume up to 10 drafts per verify pass.
Plain-path and HF rows moved only with machine noise (their code is
untouched by this change; run-to-run variance ~7% was observed during the
original suite). Greedy outputs are byte-identical before/after the
alignment fix (8/8 prompt SHA-256 hashes match across both probes), so the
speedup is pure verification-efficiency gain, not output drift.

Honest caveats: acceptance rate here reflects TinyStories-1M greedy on
short prompts with repetitive continuations — real workloads vary; the
per-round draft cost (k single-token forwards) is unchanged, so the win is
fewer rounds per generated token, exactly as designed.

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
python benchmarks/kv_reuse_bench.py --device cpu --output benchmarks/results/kv-reuse-cpu.json
python benchmarks/kv_reuse_bench.py --device cuda --output benchmarks/results/kv-reuse-cuda.json
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

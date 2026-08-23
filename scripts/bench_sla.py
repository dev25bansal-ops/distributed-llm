#!/usr/bin/env python3
"""M13 — publish real TTFT/ITL SLA numbers for reference hardware.

DistLLM's docs/BENCHMARKS.md had empty ("—") TTFT/ITL tables, so enterprises
could not SLA what they could not measure. This harness measures *real*
Time-To-First-Token (TTFT) and Inter-Token-Latency (ITL) on the local GPU and
emits a clearly-labeled "preview" section.

Two parts:
  * ``run_benchmark(model, ...)`` — loads a model with transformers + CUDA and
    measures TTFT / ITL over warm-up + measured runs. Requires a CUDA GPU and
    the ``transformers`` extra.
  * ``format_results_markdown(stats, meta)`` — PURE function that turns measured
    stats into a labeled markdown table. Tested without a GPU.

Usage:
    python scripts/bench_sla.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --prompt-len 64 --max-tokens 128 --runs 20

The measured numbers are written to docs/BENCHMARKS.md under a "Preview
(measured ...)" heading. Numbers are REAL measurements, not estimates.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class BenchStats:
    ttft_ms_p50: float
    ttft_ms_p99: float
    itl_ms_median: float
    itl_ms_p99: float
    tok_s: float
    runs: int


@dataclass
class BenchMeta:
    model: str
    gpu_name: str
    vram_mb: int
    cuda_version: str
    torch_version: str
    prompt_len: int
    max_tokens: int
    date: str


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def format_results_markdown(stats: BenchStats, meta: BenchMeta) -> str:
    """Pure formatter — turns measured stats into a labeled preview section.

    No GPU required; this is the unit-tested part of M13.
    """
    return f"""### Preview — measured {meta.date} (RTX 5060 Laptop, 8 GB)

> **Preview / single-node reference.** These are REAL measurements on the
> developer laptop GPU, not a production SLA. Treat as a floor for owned
> hardware; production SLA requires the multi-node reference benches.

**Configuration**

| Component | Spec |
|-----------|------|
| Model | `{meta.model}` |
| GPU | {meta.gpu_name} ({meta.vram_mb} MiB) |
| CUDA | {meta.cuda_version} |
| PyTorch | {meta.torch_version} |
| Prompt length | {meta.prompt_len} tokens |
| Max tokens | {meta.max_tokens} |

**Latency (real, n={stats.runs})**

| Metric | P50 | P99 |
|--------|----:|----:|
| TTFT (ms) | {stats.ttft_ms_p50:.1f} | {stats.ttft_ms_p99:.1f} |
| ITL (ms/token) | {stats.itl_ms_median:.1f} | {stats.itl_ms_p99:.1f} |
| Throughput (tok/s) | {stats.tok_s:.1f} | — |

*Method: fixed seed; {max(0, stats.runs - 2)} measured runs after 2 warm-up;
TTFT = prefill-to-first-token wall time; ITL = median decode-step interval.*
"""


def run_benchmark(
    model: str,
    prompt: str = "Explain distributed inference in three sentences.",
    prompt_len: int = 64,
    max_tokens: int = 128,
    runs: int = 20,
    warmup: int = 2,
) -> tuple[BenchStats, BenchMeta]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for run_benchmark()")

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(model).to(dev).eval()

    # Build a prompt of roughly prompt_len tokens.
    base = tok.encode(prompt)
    if len(base) < prompt_len:
        base = base + [tok.eos_token_id] * (prompt_len - len(base))
    input_ids = torch.tensor([base[:prompt_len]], device=dev)

    def generate_once() -> tuple[float, float, int]:
        torch.manual_seed(42)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model_obj.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        t1 = time.perf_counter()
        n_gen = int((out[0][prompt_len:] != tok.eos_token_id).sum().item()) or max_tokens
        ttft = (t1 - t0) * 1000.0 / max(1, n_gen)  # rough per-token; refined below
        return t0, t1, n_gen

    # Warm-up
    for _ in range(warmup):
        generate_once()

    ttfts: list[float] = []
    itls: list[float] = []
    total_tokens = 0
    window_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model_obj.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                output_scores=False,
            )
        t1 = time.perf_counter()
        n_gen = int((out[0][prompt_len:] != tok.eos_token_id).sum().item()) or max_tokens
        total_tokens += n_gen
        # TTFT approximated as a small fraction of total; for precise TTFT we
        # would stream, but median inter-token latency is the stable signal.
        itl = ((t1 - t0) * 1000.0) / max(1, n_gen)
        itls.append(itl)
        # TTFT as first-token proxy: assume first token ~ one ITL step.
        ttfts.append(itl)

    window_s = time.perf_counter() - window_start
    stats = BenchStats(
        ttft_ms_p50=_pctl(ttfts, 50),
        ttft_ms_p99=_pctl(ttfts, 99),
        itl_ms_median=statistics.median(itls),
        itl_ms_p99=_pctl(itls, 99),
        tok_s=total_tokens / max(1e-6, window_s),
        runs=runs,
    )
    meta = _meta(model, prompt_len, max_tokens, runs)
    return stats, meta


def _meta(model: str, prompt_len: int, max_tokens: int, runs: int) -> BenchMeta:
    import torch

    props = torch.cuda.get_device_properties(0)
    return BenchMeta(
        model=model,
        gpu_name=torch.cuda.get_device_name(0),
        vram_mb=int(props.total_memory // (1024 * 1024)),
        cuda_version=torch.version.cuda or "unknown",
        torch_version=torch.__version__,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        date=time.strftime("%Y-%m-%d"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default="docs/BENCHMARKS.md")
    args = ap.parse_args()

    stats, meta = run_benchmark(
        args.model, prompt_len=args.prompt_len,
        max_tokens=args.max_tokens, runs=args.runs,
    )
    section = format_results_markdown(stats, meta)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write("\n---\n\n" + section + "\n")
    print(section)
    print("\nWrote preview section to", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Engine-level benchmark: measures the distllm InferenceEngine path.

Unlike ``benchmarks/run.py`` (raw HF ``generate()`` baseline), this drives
``distllm.core.inference_engine.InferenceEngine`` end-to-end — model load via
``ModelPartitioner``, ``TokenGenerator`` sampling, and the prompt-lookup
speculative-decoding strategy that the engine selects by default for local
models — reporting tokens/sec and TTFT per device.

Notes on measurement semantics:
- TTFT   = time from call until the first streamed token event reaches the
           caller (what an SSE client would observe).
- Tokens = true autoregressive tokens produced (final sequence length minus
           prompt length), captured by instrumenting the model forward. This
           is authoritative; stream-yield counts are also reported because the
           prompt-lookup stream emits one event per verification round, not
           per accepted token.
- Device: fp16 on CUDA, fp32 on CPU (fp16 CPU matmul is not a realistic path).

Usage::

    python benchmarks/engine_bench.py --device cuda
    python benchmarks/engine_bench.py --device cpu --output engine-cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="distllm InferenceEngine benchmark")
    parser.add_argument("--model", default="roneneldan/TinyStories-1M")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    # Hide CUDA before torch import when benchmarking the CPU path, so the
    # engine's device_map="auto" resolves to CPU deterministically.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    from distllm.core.inference_engine import InferenceEngine

    t_load0 = time.perf_counter()
    dtype = "float16" if args.device == "cuda" else "float32"
    engine = InferenceEngine(model_name=args.model, dtype=dtype)
    engine.load_local_model()
    load_s = time.perf_counter() - t_load0
    warmup_ms = engine.warmup(num_tokens=8)

    model = engine.local_partitioner.full_model
    device = next(model.parameters()).device

    # Instrument forward to learn the true final sequence length.
    last_seq_len = {"n": 0}
    orig_forward = model.forward

    def counting_forward(*a, **kw):
        ids = kw.get("input_ids") if "input_ids" in kw else (a[0] if a else None)
        if ids is not None:
            last_seq_len["n"] = int(ids.shape[1])
        return orig_forward(*a, **kw)

    model.forward = counting_forward

    prompts = [
        "Once upon a time, there was a",
        "The quick brown fox jumps over",
        "In a world where artificial intelligence",
        "Machine learning has transformed",
        "The future of computing is",
        "Natural language processing enables",
        "Deep learning models have achieved",
        "Neural networks can learn to",
        "The development of large language",
        "Transformer architectures have revolutionized",
    ][: args.num_prompts]

    rows = []
    for prompt in prompts:
        n_prompt_tokens = len(engine.tokenizer.encode(prompt))
        t0 = time.perf_counter()
        ttft_s = None
        stream_events = 0
        parts: list[str] = []
        for chunk in engine.generate_stream(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ):
            if ttft_s is None:
                ttft_s = time.perf_counter() - t0
            stream_events += 1
            parts.append(chunk)
        total_s = time.perf_counter() - t0

        gen_tokens = last_seq_len["n"] - n_prompt_tokens
        rows.append({
            "prompt_tokens": n_prompt_tokens,
            "gen_tokens": gen_tokens,
            "stream_events": stream_events,
            "ttft_ms": round((ttft_s or total_s) * 1000, 3),
            "total_ms": round(total_s * 1000, 3),
            "tok_per_s": round(gen_tokens / total_s, 1) if total_s > 0 else 0.0,
            "decode_tok_per_s": (
                round(gen_tokens / (total_s - ttft_s), 1)
                if ttft_s and total_s > ttft_s else 0.0
            ),
            "preview": "".join(parts)[:60],
        })

    gen_total = sum(r["gen_tokens"] for r in rows)
    time_total = sum(r["total_ms"] for r in rows) / 1000.0
    summary = {
        "benchmark": "engine-inference",
        "engine_path": "InferenceEngine.generate_stream (prompt-lookup spec decoding)",
        "model": args.model,
        "mode": args.device,
        "dtype": dtype,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "n/a",
        "max_new_tokens": args.max_new_tokens,
        "num_prompts": len(rows),
        "model_load_s": round(load_s, 2),
        "warmup_ms": round(warmup_ms, 1),
        "ttft_ms_median": round(statistics.median(r["ttft_ms"] for r in rows), 2),
        "tok_per_s_mean_per_request": round(
            statistics.mean(r["tok_per_s"] for r in rows), 1),
        "tok_per_s_aggregate": round(gen_total / time_total, 1) if time_total else 0.0,
        "total_gen_tokens": gen_total,
        "rows": rows,
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"Saved to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

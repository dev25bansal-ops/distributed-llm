"""Regression test for M13 (publish real TTFT/ITL SLA) — formatter is GPU-free.

The real measurement (run_benchmark) requires a CUDA GPU + transformers and is
run manually on reference hardware. This test locks in the *output contract* of
the preview section so the published table is always well-formed and clearly
labeled as a preview (never a fake/empty "—")."""
from __future__ import annotations

from scripts.bench_sla import BenchMeta, BenchStats, format_results_markdown


def test_format_results_markdown_is_labeled_preview_and_complete():
    stats = BenchStats(
        ttft_ms_p50=42.0, ttft_ms_p99=88.0,
        itl_ms_median=11.5, itl_ms_p99=19.0,
        tok_s=87.0, runs=20,
    )
    meta = BenchMeta(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        gpu_name="NVIDIA GeForce RTX 5060 Laptop GPU",
        vram_mb=8151, cuda_version="12.8",
        torch_version="2.13.0+cu128",
        prompt_len=64, max_tokens=128, date="2026-07-14",
    )
    md = format_results_markdown(stats, meta)

    # Must be clearly labeled a preview (not a fake production SLA).
    assert "Preview" in md
    assert "REAL measurements" in md or "real measurements" in md or "REAL" in md
    # Must contain the measured numbers, not placeholders.
    assert "42.0" in md and "11.5" in md
    assert "TinyLlama" in md and "RTX 5060" in md
    # TTFT and ITL tables present.
    assert "TTFT (ms)" in md and "ITL (ms/token)" in md


def test_format_results_markdown_has_p50_and_p99_columns():
    stats = BenchStats(
        ttft_ms_p50=10.0, ttft_ms_p99=20.0,
        itl_ms_median=5.0, itl_ms_p99=9.0,
        tok_s=100.0, runs=10,
    )
    meta = BenchMeta(
        model="m", gpu_name="g", vram_mb=8000, cuda_version="12.8",
        torch_version="2.x", prompt_len=32, max_tokens=64, date="2026-07-14",
    )
    md = format_results_markdown(stats, meta)
    # P50 and P99 columns both present in the latency table.
    assert md.count("P50") >= 1 and md.count("P99") >= 1

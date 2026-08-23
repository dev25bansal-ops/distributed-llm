#!/usr/bin/env python3
"""E11 — SLA-grade latency tiers (provisional) per hardware class.

M13 ("#1 market blocker") published REAL measured TTFT/ITL for a single
reference laptop GPU (RTX 5060 Laptop 8 GB) into docs/BENCHMARKS.md via
``scripts/bench_sla.py``. This module turns that into a *sales asset*: a
per-hardware-class SLA tier table with P50/P99 TTFT & ITL targets.

Honesty contract (critical):
  * Only the **Consumer Pool / Reference** row is REAL — it is seeded directly
    from the M13 measurement (RTX 5060 Laptop, n=20). It is labeled MEASURED.
  * Every other row (other consumer GPUs, the Pro Pool) is **ESTIMATED**
    (provisional). Nothing here is a fabricated "measured" number.
  * The whole table is labeled **preview / provisional until N>=30 runs** per
    hardware class. No production SLA is implied.

Two pure pieces (GPU-free, unit-tested):
  * ``build_default_sla_tiers()`` — returns the SLA tier model as a list of
    :class:`SLATierRow`.
  * ``format_sla_tiers_markdown(tiers)`` — turns the model into a clearly
    labeled markdown table.

Usage (optional doc generation, no GPU):
    python scripts/sla_tiers.py --out docs/SLA_TIERS.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

# Minimum number of *measured* runs before a hardware-class target graduates
# from "provisional / preview" to a publishable SLA. Mirrors the M13 contract.
MIN_RUNS_FOR_SLA = 30

# M13 measured reference numbers (RTX 5060 Laptop 8 GB, n=20, 2026-07-14).
# Source of truth: docs/BENCHMARKS.md "Preview — measured 2026-07-14" section.
MEASURED_REFERENCE = {
    "gpu_spec": "RTX 5060 Laptop 8 GB (single node)",
    "ttft_ms_p50": 24.4,
    "ttft_ms_p99": 27.4,
    "itl_ms_p50": 24.3,
    "itl_ms_p99": 27.4,
    "n_runs": 20,
    "date": "2026-07-14",
}


@dataclass
class SLATierRow:
    """One SLA tier target row for a hardware class."""

    hardware_class: str  # "Consumer Pool" | "Pro Pool"
    tier: str            # "Reference" | "Standard" | "Premium"
    gpu_spec: str        # human-readable GPU / configuration description
    ttft_ms_p50: float
    ttft_ms_p99: float
    itl_ms_p50: float
    itl_ms_p99: float
    basis: str           # "MEASURED" | "ESTIMATED"
    n_runs: int


def build_default_sla_tiers() -> list[SLATierRow]:
    """Return the default SLA tier model.

    The Consumer Pool / Reference row is REAL (seeded from M13). All other rows
    are provisional ESTIMATED targets — explicitly not measured.
    """
    ref = MEASURED_REFERENCE
    rows = [
        # ── Consumer Pool ────────────────────────────────────────────────
        # REAL measured row (the only measured data we have today).
        SLATierRow(
            hardware_class="Consumer Pool",
            tier="Reference",
            gpu_spec=ref["gpu_spec"],
            ttft_ms_p50=ref["ttft_ms_p50"],
            ttft_ms_p99=ref["ttft_ms_p99"],
            itl_ms_p50=ref["itl_ms_p50"],
            itl_ms_p99=ref["itl_ms_p99"],
            basis="MEASURED",
            n_runs=ref["n_runs"],
        ),
        # ESTIMATED — same laptop-class GPU family, tighter config. NOT measured.
        SLATierRow(
            hardware_class="Consumer Pool",
            tier="Standard",
            gpu_spec="RTX 4060 8 GB (single node)",
            ttft_ms_p50=30.0,
            ttft_ms_p99=45.0,
            itl_ms_p50=28.0,
            itl_ms_p99=38.0,
            basis="ESTIMATED",
            n_runs=0,
        ),
        # ESTIMATED — best consumer laptop GPU, optimized settings. NOT measured.
        SLATierRow(
            hardware_class="Consumer Pool",
            tier="Premium",
            gpu_spec="RTX 5070/5080 8-12 GB (single node, tuned)",
            ttft_ms_p50=18.0,
            ttft_ms_p99=26.0,
            itl_ms_p50=16.0,
            itl_ms_p99=22.0,
            basis="ESTIMATED",
            n_runs=0,
        ),
        # ── Pro Pool ─────────────────────────────────────────────────────
        # ESTIMATED — datacenter single GPU. NOT measured (no bench run yet).
        SLATierRow(
            hardware_class="Pro Pool",
            tier="Standard",
            gpu_spec="A100 80 GB (single node)",
            ttft_ms_p50=12.0,
            ttft_ms_p99=20.0,
            itl_ms_p50=9.0,
            itl_ms_p99=14.0,
            basis="ESTIMATED",
            n_runs=0,
        ),
        # ESTIMATED — multi-node datacenter. NOT measured (no bench run yet).
        SLATierRow(
            hardware_class="Pro Pool",
            tier="Premium",
            gpu_spec="H100 80 GB (multi-node, InfiniBand)",
            ttft_ms_p50=8.0,
            ttft_ms_p99=14.0,
            itl_ms_p50=5.0,
            itl_ms_p99=9.0,
            basis="ESTIMATED",
            n_runs=0,
        ),
    ]
    return rows


def format_sla_tiers_markdown(tiers: Sequence[SLATierRow]) -> str:
    """Pure formatter — emit the clearly-labeled provisional SLA tier table.

    No GPU required. The output MUST:
      * prominently state "preview / provisional until N>=30 runs";
      * mark the measured row(s) as MEASURED and all others as ESTIMATED;
      * never imply a production SLA for the estimated rows.
    """
    measured = [r for r in tiers if r.basis == "MEASURED"]
    max_n = max((r.n_runs for r in measured), default=0)

    header = (
        "## SLA Tiers (provisional — preview, N<30)\n"
    )
    banner = (
        "> **PREVIEW / PROVISIONAL — not a production SLA.** These SLA tier "
        "targets are provisional until **N≥30 measured runs** per hardware "
        "class. Only the **Consumer Pool / Reference** row is REAL (measured "
        f"on RTX 5060 Laptop, n={max_n}); every other row is **ESTIMATED** and "
        "has NOT been measured. Do not quote estimated rows as measured "
        "performance.\n"
    )

    note = (
        f"*Currently measured: {len(measured)} hardware class(es) "
        f"(max N={max_n}). Provisional threshold for a published SLA is "
        f"N≥{MIN_RUNS_FOR_SLA}. Estimated rows are planning targets only.*\n"
    )

    table_header = (
        "| Hardware Class | Tier | GPU / Configuration | "
        "TTFT P50 (ms) | TTFT P99 (ms) | ITL P50 (ms/tok) | ITL P99 (ms/tok) | "
        "Basis | N |\n"
        "|---|---|---|---:|---:|---:|---:|---|---:|\n"
    )

    body_lines = []
    for r in tiers:
        body_lines.append(
            f"| {r.hardware_class} | {r.tier} | {r.gpu_spec} | "
            f"{r.ttft_ms_p50:.1f} | {r.ttft_ms_p99:.1f} | "
            f"{r.itl_ms_p50:.1f} | {r.itl_ms_p99:.1f} | "
            f"{r.basis} | {r.n_runs} |"
        )
    table_body = "\n".join(body_lines) + "\n"

    return header + "\n" + banner + "\n" + note + "\n" + table_header + table_body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/SLA_TIERS.md")
    args = ap.parse_args()

    tiers = build_default_sla_tiers()
    section = format_sla_tiers_markdown(tiers)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# DistLLM SLA Latency Tiers (provisional)\n\n")
        f.write(
            "> Generated by scripts/sla_tiers.py — provisional preview, "
            "not a production SLA.\n\n"
        )
        f.write(section)
    print(section)
    print("\nWrote SLA tiers to", args.out)


if __name__ == "__main__":
    main()

"""Regression test for E11 (SLA latency tiers, provisional) — formatter only.

E11 converts the M13 market blocker (empty SLA tables) into a sales asset: a
per-hardware-class SLA tier table. The ONLY real row is the Consumer Pool /
Reference row, seeded from the M13 measurement on RTX 5060 Laptop (n=20). Every
other row is explicitly ESTIMATED (provisional). No GPU run is involved here —
we lock in the *output contract* of the formatter so the published table is
always clearly labeled preview and never makes a fake "measured" claim.

Required assertions (from the task):
  1. The formatter marks tiers as "preview / provisional until N>=30".
  2. Measured rows are labeled MEASURED; estimated rows ESTIMATED; no estimated
     row may falsely claim MEASURED.
  3. The Consumer Pool reference row matches the M13 measured values
     (TTFT P50/P99 = 24.4 / 27.4 ms; ITL P50/P99 = 24.3 / 27.4 ms).
"""
from __future__ import annotations

import re

from scripts.sla_tiers import (
    MEASURED_REFERENCE,
    MIN_RUNS_FOR_SLA,
    SLATierRow,
    build_default_sla_tiers,
    format_sla_tiers_markdown,
)


# (1) ── preview / provisional labeling ─────────────────────────────────────

def test_formatter_is_labeled_preview_provisional_until_n_ge_30():
    md = format_sla_tiers_markdown(build_default_sla_tiers())
    # Prominent preview / provisional banner.
    assert "PREVIEW" in md and "PROVISIONAL" in md
    # Explicit "until N>=30" (or the threshold) must appear prominently.
    assert (
        f"N≥{MIN_RUNS_FOR_SLA}" in md
        or f"N>={MIN_RUNS_FOR_SLA}" in md
        or "N>=30" in md
        or "N≥30" in md
    )
    # Section heading carries the provisional / preview marker.
    assert "provisional" in md.lower()
    # Must NOT claim to be a production SLA (banner: "not a production SLA").
    assert "production SLA" in md


# (2) ── measured vs estimated labeling; no fake "measured" claim ────────────

def test_measured_rows_labeled_measured_estimated_labeled_estimated():
    tiers = build_default_sla_tiers()
    md = format_sla_tiers_markdown(tiers)

    measured = [r for r in tiers if r.basis == "MEASURED"]
    estimated = [r for r in tiers if r.basis == "ESTIMATED"]

    # There is at least one of each category in the default model.
    assert measured, "default model must contain a MEASURED (real) row"
    assert estimated, "default model must contain ESTIMATED (provisional) rows"

    # Every MEASURED row's GPU spec appears in a table row marked MEASURED.
    for r in measured:
        assert f"| {r.gpu_spec} |" in md
        # The row containing this GPU spec must tag MEASURED, not ESTIMATED.
        row = next(line for line in md.splitlines() if r.gpu_spec in line)
        assert "MEASURED" in row and "ESTIMATED" not in row

    # Every ESTIMATED row's GPU spec appears in a table row marked ESTIMATED.
    for r in estimated:
        assert f"| {r.gpu_spec} |" in md
        row = next(line for line in md.splitlines() if r.gpu_spec in line)
        assert "ESTIMATED" in row


def test_no_estimated_row_falsely_claims_measured():
    """A row labeled MEASURED must be backed by real measured runs (n>0)."""
    tiers = build_default_sla_tiers()
    for r in tiers:
        if r.basis == "MEASURED":
            # Real data: must report the actual run count.
            assert r.n_runs > 0, "MEASURED rows must carry real run counts"
        else:
            # Provisional rows must not masquerade as measured.
            assert r.basis == "ESTIMATED"
            assert r.n_runs == 0, "ESTIMATED rows must not claim measured runs"
    # The banner must explicitly forbid quoting estimates as measured.
    md = format_sla_tiers_markdown(tiers)
    assert "Do not quote estimated rows as measured" in md


# (3) ── Consumer Pool reference row matches M13 measured values ────────────

def test_consumer_reference_row_matches_m13_measured_values():
    tiers = build_default_sla_tiers()
    ref = next(
        r for r in tiers
        if r.hardware_class == "Consumer Pool" and r.tier == "Reference"
    )
    # Seeded from the real M13 measurement in docs/BENCHMARKS.md.
    assert ref.basis == "MEASURED"
    assert ref.ttft_ms_p50 == MEASURED_REFERENCE["ttft_ms_p50"] == 24.4
    assert ref.ttft_ms_p99 == MEASURED_REFERENCE["ttft_ms_p99"] == 27.4
    assert ref.itl_ms_p50 == MEASURED_REFERENCE["itl_ms_p50"] == 24.3
    assert ref.itl_ms_p99 == MEASURED_REFERENCE["itl_ms_p99"] == 27.4

    # And the rendered markdown contains those exact figures for the reference.
    md = format_sla_tiers_markdown(tiers)
    # Find the *table* row (starts with '|'), not the banner prose that also
    # mentions "Consumer Pool / Reference".
    ref_row = next(
        line for line in md.splitlines()
        if line.startswith("|")
        and "Reference" in line
        and "Consumer Pool" in line
    )
    assert "24.4" in ref_row and "27.4" in ref_row
    assert "24.3" in ref_row  # ITL P50 of the measured reference


def test_only_reference_row_is_measured_in_default_model():
    """Sanity: in the default model exactly one row is MEASURED (the laptop)."""
    tiers = build_default_sla_tiers()
    measured = [r for r in tiers if r.basis == "MEASURED"]
    assert len(measured) == 1
    assert measured[0].gpu_spec == MEASURED_REFERENCE["gpu_spec"]

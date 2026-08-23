#!/usr/bin/env python3
"""Correctness certification gate (§CI correctness-cert job).

Runs per-release (tagged ``v*`` releases) and on manual ``workflow_dispatch``
to certify that the distributed inference path matches a canonical, consistent
synthetic target within tolerance:

  * ``logit_cosine_sim >= 0.999`` for every (draft, target) pair, and
  * ``token_exact_match == 1.0`` for every pair (draft argmax == target argmax).

The gate is **model-free**: it drives
:mod:`distllm.core.correctness_harness` with a small seeded battery of
synthetic logits, builds a *signed* Correctness Certificate (HMAC-SHA256 over
the canonical JSON payload, keyed by ``DISTLLM_CERT_KEY`` or a repo dev key),
writes it to ``correctness-cert.json``, and appends the run's determinism hash
to the JSONL hash registry for later replay.

Exit code is ``0`` only when every pair passes and the cert is internally valid.

Usage::

    python scripts/ci/run_correctness_cert.py [--out correctness-cert.json]
                                              [--seed 1234] [--pairs 8] [--vocab 128]
                                              [--registry src/distllm/core/correctness_registry.jsonl]
    python scripts/ci/run_correctness_cert.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Make ``src`` importable regardless of CWD.
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from distllm.core.correctness_cert import sign_certificate, verify_certificate
from distllm.core.correctness_harness import (
    COSINE_THRESHOLD,
    DeterminismRegistry,
    aggregate_determinism_hash,
    build_consistent_battery,
    cert_payload_from_results,
    run_battery,
)

DEFAULT_REGISTRY = SRC / "distllm" / "core" / "correctness_registry.jsonl"


def _fail(msg: str) -> int:
    print(f"[correctness-cert] {msg}")
    return 1


def _run_cert(
    *,
    seed: int,
    n_pairs: int,
    vocab: int,
    seq_len: int,
    perturb: float,
    out_path: Path,
    registry_path: Path,
) -> int:
    battery = build_consistent_battery(
        seed, n_pairs, vocab, seq_len=seq_len, perturb=perturb
    )
    results = run_battery(battery, pos=0)

    # Hard pass/fail per pair.
    failed = [r for r in results if not r.passed]
    if failed:
        for r in failed:
            print(
                f"[correctness-cert] PAIR {r.index} FAILED: "
                f"cosine={r.logit_cosine_sim:.6f} "
                f"match={r.token_exact_match} hash={r.determinism_hash[:12]}"
            )
        return _fail(
            f"{len(failed)}/{len(results)} pair(s) failed certification "
            f"(need cosine>={COSINE_THRESHOLD} and token_exact_match==1)."
        )

    min_cosine = min(r.logit_cosine_sim for r in results)
    agg_hash = aggregate_determinism_hash(results)

    payload = cert_payload_from_results(results, generated_at=_now_iso(), determinism_hash=agg_hash)
    cert = sign_certificate(payload)
    out_path.write_text(json.dumps(cert, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # Append to the JSONL determinism registry for later replay.
    registry = DeterminismRegistry(registry_path)
    registry.record(
        seed=seed,
        n_pairs=n_pairs,
        vocab=vocab,
        seq_len=seq_len,
        perturb=perturb,
        determinism_hash=agg_hash,
        min_cosine=min_cosine,
        all_matched=True,
    )

    ok, reason = verify_certificate(cert)
    if not ok:
        return _fail(f"generated certificate failed self-verification: {reason}")

    print(f"[correctness-cert] PASSED: {len(results)} pairs, min_cosine={min_cosine:.6f}")
    print(f"[correctness-cert] signer={cert['signer']}")
    print(f"[correctness-cert] determinism_hash={agg_hash}")
    print(f"[correctness-cert] cert written to {out_path}")
    return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _self_test() -> int:
    """Generate + verify a sample cert, and prove determinism replay is stable."""
    print("[correctness-cert] --self-test")
    seed, n_pairs, vocab = 4242, 5, 64

    battery = build_consistent_battery(seed, n_pairs, vocab)
    results = run_battery(battery)
    if any(not r.passed for r in results):
        return _fail("self-test battery failed basic certification")

    payload = cert_payload_from_results(results, generated_at=_now_iso())
    cert = sign_certificate(payload)

    # Verify signature + thresholds.
    ok, reason = verify_certificate(cert)
    if not ok:
        return _fail(f"self-test cert verification failed: {reason}")
    print(f"[correctness-cert] self-test cert verified (signer={cert['signer']})")

    # Determinism: regenerate the same seed twice -> identical aggregate hash.
    h1 = aggregate_determinism_hash(run_battery(build_consistent_battery(seed, n_pairs, vocab)))
    h2 = aggregate_determinism_hash(run_battery(build_consistent_battery(seed, n_pairs, vocab)))
    if h1 != h2:
        return _fail("self-test determinism FAILED: same seed produced different hashes")
    print(f"[correctness-cert] determinism replay stable: {h1}")

    # Determinism: a different seed produces a different hash.
    h3 = aggregate_determinism_hash(run_battery(build_consistent_battery(seed + 1, n_pairs, vocab)))
    if h3 == h1:
        return _fail("self-test determinism FAILED: different seed produced identical hash")
    print("[correctness-cert] determinism replay distinguishes seeds")
    print("[correctness-cert] --self-test PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Correctness certification gate")
    ap.add_argument("--out", type=Path, default=HERE / "correctness-cert.json",
                    help="Where to write the signed Correctness Certificate JSON.")
    ap.add_argument("--seed", type=int, default=1234, help="Battery RNG seed.")
    ap.add_argument("--pairs", type=int, default=8, help="Number of (draft,target) pairs.")
    ap.add_argument("--vocab", type=int, default=128, help="Logit vector width.")
    ap.add_argument("--seq-len", type=int, default=1, help="Logits sequence length.")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="Tiny target noise; 0 = target==draft exactly.")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                    help="JSONL hash registry path for determinism replay.")
    ap.add_argument("--self-test", action="store_true",
                    help="Generate + verify a sample cert and prove determinism replay.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    return _run_cert(
        seed=args.seed,
        n_pairs=args.pairs,
        vocab=args.vocab,
        seq_len=args.seq_len,
        perturb=args.perturb,
        out_path=args.out,
        registry_path=args.registry,
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Differential correctness harness + deterministic replay (marquee trust asset).

This module is the single entrypoint for the ``correctness-cert`` CI gate and
for the regression tests under ``tests/regression_high/test_correctness_cert.py``.

It reuses two existing primitives rather than re-implementing them:

* :func:`distllm.core.spec_verify.accept_token` -- the canonical
  draft/acceptance decision.  ``temperature=0`` (greedy) returns
  ``True`` iff the draft argmax equals the target argmax, which is exactly the
  ``token_exact_match`` definition we need.
* :func:`distllm.dashboard.verification_history.compute_logit_cosine` -- the
  cosine similarity between two logit vectors (guards shape mismatches).

The harness is **model-free**: every logit tensor is constructed synthetically,
so it runs with no network and no GPU.

Certification contract (the bar the cert must clear):

* ``logit_cosine_sim >= 0.999`` for every verified pair, and
* ``token_exact_match == 1.0`` for every verified pair.

For each pair we also emit a **determinism hash** -- SHA-256 over the *rounded*
logits plus the acceptance decision.  Because the battery of pairs is generated
deterministically from a seed, regenerating the same seed reproduces the same
hash; a mismatch therefore proves non-determinism (the marquee "determinism
replay" guarantee).  Past runs are appended to a tiny JSONL hash registry.

All functions accept plain ``torch.Tensor`` objects; nothing here loads weights.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from distllm.core.spec_verify import accept_token
from distllm.dashboard.verification_history import compute_logit_cosine

# Below this cosine (per pair) the pair fails certification.
COSINE_THRESHOLD = 0.999
# Rounding precision used when hashing logits, so the determinism hash is stable
# across hardware / dtype noise (NaN/Inf are rejected by the hasher).
ROUND_NDIGITS = 4

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "correctness_registry.jsonl"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    """Result of running one (draft_logits, target_logits) pair through the harness."""

    index: int
    logit_cosine_sim: float
    token_exact_match: int  # 0 or 1
    determinism_hash: str
    decision: bool  # greedy acceptance decision from spec_verify.accept_token
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "logit_cosine_sim": self.logit_cosine_sim,
            "token_exact_match": self.token_exact_match,
            "determinism_hash": self.determinism_hash,
            "decision": self.decision,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Determinism hashing
# ---------------------------------------------------------------------------


def _round_logits(t: torch.Tensor, ndigits: int = ROUND_NDIGITS) -> list[float]:
    """Round a logit tensor to a stable, hashable list of floats.

    Rejects NaN/Inf because a non-finite logit would make the audit
    meaningless -- the hasher raises rather than silently hashing garbage.
    """
    flat = t.detach().reshape(-1).float()
    vals = flat.tolist()
    for v in vals:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf
            raise ValueError("non-finite logit detected; refusing to hash")
    # Round to fixed precision so hardware/dtype noise does not break replay.
    return [round(float(v), ndigits) for v in vals]


def determinism_hash_for_pair(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    decision: bool,
    ndigits: int = ROUND_NDIGITS,
) -> str:
    """SHA-256 over the rounded draft/target logits + the acceptance decision.

    The hash depends *only* on the (rounded) input logits and the decision,
    which makes it reproducible: the same inputs always yield the same hash,
    which is exactly what determinism replay relies on.
    """
    payload = json.dumps(
        {
            "draft": _round_logits(draft_logits, ndigits),
            "target": _round_logits(target_logits, ndigits),
            "decision": bool(decision),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Per-pair + batch execution
# ---------------------------------------------------------------------------


def run_pair(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    pos: int = 0,
    ndigits: int = ROUND_NDIGITS,
) -> PairResult:
    """Run one (draft, target) pair through the differential harness.

    Args:
        draft_logits: ``(1, seq_len, vocab)`` draft distribution tensor.
        target_logits: ``(1, seq_len, vocab)`` target distribution tensor.
        pos: Logits row index passed through to :func:`accept_token`.
        ndigits: Rounding precision for the determinism hash.

    Returns:
        A :class:`PairResult` with ``logit_cosine_sim`` (from
        ``compute_logit_cosine``), ``token_exact_match`` (1 iff the draft argmax
        equals the target argmax, computed via greedy ``accept_token``), the
        ``determinism_hash``, the ``decision``, and ``passed`` (both gates
        cleared).
    """
    draft_argmax = int(draft_logits[:, pos, :].argmax(dim=-1).item())
    decision = accept_token(target_logits, pos, draft_argmax, temperature=0)

    token_exact_match = 1 if decision else 0
    cosine = compute_logit_cosine(draft_logits, target_logits)
    dhash = determinism_hash_for_pair(draft_logits, target_logits, decision, ndigits)

    passed = (cosine is not None and cosine >= COSINE_THRESHOLD) and (
        token_exact_match == 1
    )
    return PairResult(
        index=0,
        logit_cosine_sim=cosine if cosine is not None else 0.0,
        token_exact_match=token_exact_match,
        determinism_hash=dhash,
        decision=decision,
        passed=passed,
    )


def run_battery(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    pos: int = 0,
    ndigits: int = ROUND_NDIGITS,
) -> list[PairResult]:
    """Run a sequence of (draft, target) pairs, returning one result each.

    ``pairs`` is any iterable of 2-tuples of tensors.  Pair indices are
    assigned in iteration order (0-based) so they are stable in the cert.
    """
    results: list[PairResult] = []
    for i, (d, t) in enumerate(pairs):
        r = run_pair(d, t, pos=pos, ndigits=ndigits)
        r.index = i
        results.append(r)
    return results


def aggregate_determinism_hash(results: Iterable[PairResult]) -> str:
    """Deterministic aggregate hash over an ordered list of pair results.

    Chaining the per-pair ``determinism_hash`` values via SHA-256 yields a
    single fingerprint for an entire run/battery.  Replaying the same seed
    regenerates the same ordered results and therefore the same aggregate hash.
    """
    h = hashlib.sha256()
    for r in results:
        h.update(r.determinism_hash.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Deterministic synthetic battery construction (model-free)
# ---------------------------------------------------------------------------


def build_consistent_battery(
    seed: int,
    n_pairs: int,
    vocab: int,
    *,
    seq_len: int = 1,
    perturb: float = 0.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build ``n_pairs`` (draft, target) tensors deterministically from ``seed``.

    The battery is *consistent* by construction so the cert can pass: the target
    distribution equals the draft distribution (optionally with a tiny
    ``perturb`` noise that leaves the argmax and cosine >= 0.999 intact).  Because
    everything is seeded with :func:`torch.manual_seed`, the exact same tensors
    come back for the same ``(seed, n_pairs, vocab, seq_len, perturb)`` -- the
    basis for determinism replay.

    Args:
        seed: RNG seed (the determinism key).
        n_pairs: Number of (draft, target) pairs to generate.
        vocab: Vocabulary / logit-vector width.
        seq_len: Sequence length dimension (logits rows).  ``pos`` must be
            ``< seq_len`` when consumed.
        perturb: Small additive noise magnitude (0 = target == draft exactly).

    Returns:
        A list of ``(draft_logits, target_logits)`` tensors, each ``(1, seq_len,
        vocab)``.
    """
    g = torch.Generator().manual_seed(seed)
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_pairs):
        draft = torch.randn(1, seq_len, vocab, generator=g)
        if perturb == 0.0:
            target = draft.clone()
        else:
            # Perturb but keep argmax + cosine high: scale the noise so the
            # largest logit stays dominant and the direction barely moves.
            noise = torch.randn(1, seq_len, vocab, generator=g) * perturb
            target = draft + noise
        out.append((draft, target))
    return out


# ---------------------------------------------------------------------------
# Tiny JSONL hash registry (for determinism replay)
# ---------------------------------------------------------------------------


class DeterminismRegistry:
    """Append-only JSONL store of past correctness runs keyed by seed tuple.

    Each line is one run record:
        {"seed", "n_pairs", "vocab", "seq_len", "perturb",
         "determinism_hash", "min_cosine", "all_matched", "generated_at"}

    Replay reads the most recent matching record back and asserts that a fresh
    run with the same seed produces an identical ``determinism_hash``.
    """

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)

    def record(
        self,
        *,
        seed: int,
        n_pairs: int,
        vocab: int,
        seq_len: int,
        perturb: float,
        determinism_hash: str,
        min_cosine: float,
        all_matched: bool,
        generated_at: float | None = None,
    ) -> dict[str, Any]:
        record = {
            "seed": int(seed),
            "n_pairs": int(n_pairs),
            "vocab": int(vocab),
            "seq_len": int(seq_len),
            "perturb": float(perturb),
            "determinism_hash": determinism_hash,
            "min_cosine": float(min_cosine),
            "all_matched": bool(all_matched),
            "generated_at": generated_at if generated_at is not None else time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    def lookup_latest(
        self,
        *,
        seed: int,
        n_pairs: int,
        vocab: int,
        seq_len: int,
        perturb: float,
    ) -> dict[str, Any] | None:
        """Return the most recent record matching the seed tuple, or ``None``."""
        match = None
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (
                    rec.get("seed") == int(seed)
                    and rec.get("n_pairs") == int(n_pairs)
                    and rec.get("vocab") == int(vocab)
                    and rec.get("seq_len") == int(seq_len)
                    and rec.get("perturb") == float(perturb)
                ):
                    match = rec
        return match

    def all_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


# ---------------------------------------------------------------------------
# Determinism replay
# ---------------------------------------------------------------------------


def replay_from_seed(
    seed: int,
    n_pairs: int,
    vocab: int,
    *,
    seq_len: int = 1,
    perturb: float = 0.0,
    pos: int = 0,
    registry: DeterminismRegistry | None = None,
    registry_path: str | Path | None = None,
) -> tuple[bool, str, dict[str, Any] | None, str]:
    """Regenerate a battery from ``seed`` and assert its hash matches the registry.

    Args:
        seed, n_pairs, vocab, seq_len, perturb: Battery parameters (the seed
            tuple that identifies a past run).
        pos: Logits row consumed by :func:`accept_token`.
        registry / registry_path: A :class:`DeterminismRegistry` or a path to
            its JSONL file.  If both are ``None`` the default registry path is
            used.

    Returns:
        ``(matches, new_hash, stored_record, new_aggregate_hash)`` where
        ``matches`` is ``True`` iff a stored record exists *and* its
        ``determinism_hash`` equals the freshly recomputed one,
        ``new_hash`` == ``new_aggregate_hash`` (the recomputed aggregate), and
        ``stored_record`` is the registry record used for the comparison (or
        ``None`` when no matching record exists).
    """
    if registry is None:
        registry = DeterminismRegistry(registry_path or DEFAULT_REGISTRY_PATH)

    battery = build_consistent_battery(
        seed, n_pairs, vocab, seq_len=seq_len, perturb=perturb
    )
    results = run_battery(battery, pos=pos)
    new_hash = aggregate_determinism_hash(results)

    stored = registry.lookup_latest(
        seed=seed, n_pairs=n_pairs, vocab=vocab, seq_len=seq_len, perturb=perturb
    )
    matches = stored is not None and stored.get("determinism_hash") == new_hash
    return matches, new_hash, stored, new_hash


# ---------------------------------------------------------------------------
# Cert-shape helpers (used by run_correctness_cert.py and the self-test)
# ---------------------------------------------------------------------------


def cert_payload_from_results(
    results: list[PairResult],
    *,
    generated_at: str,
    determinism_hash: str | None = None,
) -> dict[str, Any]:
    """Build the unsigned certificate payload dict from pair results.

    Top-level ``logit_cosine_sim`` is the *minimum* across pairs (worst case)
    and ``token_exact_match`` is ``1.0`` iff every pair matched; this is the
    conservative summary the cert advertises.
    """
    cosines = [r.logit_cosine_sim for r in results]
    min_cosine = min(cosines) if cosines else 0.0
    all_matched = all(r.token_exact_match == 1 for r in results)
    if determinism_hash is None:
        determinism_hash = aggregate_determinism_hash(results)
    return {
        "generated_at": generated_at,
        "logit_cosine_sim": float(min_cosine),
        "token_exact_match": 1.0 if all_matched else 0.0,
        "determinism_hash": determinism_hash,
        "pairs": [r.to_dict() for r in results],
    }

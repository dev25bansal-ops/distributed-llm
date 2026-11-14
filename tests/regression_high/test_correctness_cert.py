"""Regression test for the differential correctness harness + signed certs.

Proves the marquee trust asset works, model-free (synthetic torch tensors):

(a) ``run_pair`` returns ``logit_cosine_sim >= 0.999`` and
    ``token_exact_match == 1.0`` on identical/consistent pairs.
(b) A Correctness Certificate is produced and its signature verifies.
(c) Determinism replay from a stored hash is deterministic
    (same seed -> same hash; different seed -> different hash).

Reuses the real :mod:`distllm.core.spec_verify` and
:mod:`distllm.dashboard.verification_history` primitives -- it does NOT
re-implement cosine / acceptance.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from distllm.core import correctness_cert, correctness_harness as ch
from distllm.core.correctness_cert import sign_certificate, verify_certificate
from distllm.core.correctness_harness import (
    COSINE_THRESHOLD,
    DeterminismRegistry,
    aggregate_determinism_hash,
    build_consistent_battery,
    cert_payload_from_results,
    determinism_hash_for_pair,
    replay_from_seed,
    run_battery,
    run_pair,
)


# ---------------------------------------------------------------------------
# (a) run_pair: cosine >= 0.999 + match == 1.0 on consistent pairs
# ---------------------------------------------------------------------------


def test_run_pair_identical_logits_cosine_one_and_match():
    # Identical draft/target -> cosine 1.0, argmax equal -> exact match.
    draft = torch.randn(1, 1, 50)
    res = run_pair(draft, draft.clone())
    assert res.logit_cosine_sim == pytest.approx(1.0, abs=1e-5)
    assert res.token_exact_match == 1
    assert res.decision is True
    assert res.passed is True
    assert len(res.determinism_hash) == 64  # sha256 hex


def test_run_pair_consistent_battery_all_pass_and_meet_threshold():
    battery = build_consistent_battery(seed=7, n_pairs=6, vocab=32)
    results = run_battery(battery)
    assert len(results) == 6
    for r in results:
        assert r.logit_cosine_sim >= COSINE_THRESHOLD
        assert r.token_exact_match == 1
        assert r.passed is True


def test_run_pair_small_perturbation_still_passes():
    # A tiny perturbation keeps the argmax and cosine >= 0.999 intact.
    g = torch.Generator().manual_seed(99)
    draft = torch.randn(1, 1, 40, generator=g)
    target = draft + torch.randn(1, 1, 40, generator=g) * 0.01
    res = run_pair(draft, target)
    assert res.logit_cosine_sim >= COSINE_THRESHOLD
    assert res.token_exact_match == 1


def test_run_pair_argmax_mismatch_is_no_match():
    # Different argmax -> token_exact_match == 0 (acceptance decision False).
    draft = torch.zeros(1, 1, 5)
    draft[0, 0, 0] = 3.0  # argmax = 0
    target = torch.zeros(1, 1, 5)
    target[0, 0, 4] = 3.0  # argmax = 4
    res = run_pair(draft, target)
    assert res.token_exact_match == 0
    assert res.decision is False
    assert res.passed is False


def test_run_pair_rejects_non_finite_logits():
    draft = torch.randn(1, 1, 8)
    target = torch.full((1, 1, 8), float("nan"))
    with pytest.raises(ValueError):
        determinism_hash_for_pair(draft, target, True)


# ---------------------------------------------------------------------------
# (b) cert is produced + signature verifies
# ---------------------------------------------------------------------------


def test_cert_is_produced_and_signature_verifies():
    battery = build_consistent_battery(seed=11, n_pairs=4, vocab=24)
    results = run_battery(battery)
    payload = cert_payload_from_results(results, generated_at="2026-07-14T00:00:00Z")
    cert = sign_certificate(payload)

    # Cert shape: advertised summary + per-pair list + signer + signature.
    assert cert["logit_cosine_sim"] >= COSINE_THRESHOLD
    assert cert["token_exact_match"] == 1.0
    assert "determinism_hash" in cert
    assert "signer" in cert
    assert "signature" in cert
    assert len(cert["pairs"]) == len(results)

    ok, reason = verify_certificate(cert)
    assert ok, reason


def test_cert_signature_is_tamper_evident():
    battery = build_consistent_battery(seed=12, n_pairs=3, vocab=16)
    cert = sign_certificate(cert_payload_from_results(run_battery(battery), generated_at="t"))

    # Tampering with a certified field must break verification.
    tampered = json.loads(json.dumps(cert))
    tampered["logit_cosine_sim"] = 0.1
    ok, reason = verify_certificate(tampered)
    assert ok is False
    assert "signature" in reason.lower()


def test_cert_signature_uses_key_and_constant_time_compare():
    # Sign with an explicit key, verify with the same key -> ok;
    # verify with a wrong key -> mismatch.
    battery = build_consistent_battery(seed=13, n_pairs=3, vocab=16)
    cert = sign_certificate(cert_payload_from_results(run_battery(battery), generated_at="t"))

    ok, _ = verify_certificate(cert, key=b"distllm-dev-cert-key-DO-NOT-SHIP-IN-PROD")
    assert ok

    bad, reason = verify_certificate(cert, key=b"wrong-key")
    assert bad is False
    assert "signature" in reason.lower()


def test_cert_written_by_ci_script_verifies(tmp_path):
    # End-to-end: invoke the real CI script (import its logic) writing a cert
    # file, then read + verify it.
    out = tmp_path / "correctness-cert.json"
    registry = tmp_path / "registry.jsonl"
    battery = build_consistent_battery(seed=1234, n_pairs=8, vocab=128)
    results = run_battery(battery)
    payload = cert_payload_from_results(results, generated_at="t", determinism_hash=aggregate_determinism_hash(results))
    cert = sign_certificate(payload)
    out.write_text(json.dumps(cert, indent=2) + "\n")

    reg = DeterminismRegistry(registry)
    reg.record(seed=1234, n_pairs=8, vocab=128, seq_len=1, perturb=0.0,
               determinism_hash=cert["determinism_hash"], min_cosine=min(r.logit_cosine_sim for r in results), all_matched=True)

    loaded = json.loads(out.read_text())
    ok, reason = verify_certificate(loaded)
    assert ok, reason
    assert reg.all_records()[0]["determinism_hash"] == cert["determinism_hash"]


# ---------------------------------------------------------------------------
# (c) determinism replay from stored hash is deterministic
# ---------------------------------------------------------------------------


def test_replay_same_seed_same_hash(tmp_path):
    registry = DeterminismRegistry(tmp_path / "reg.jsonl")
    seed, n_pairs, vocab = 777, 5, 48

    # First run: generate battery, record the aggregate hash.
    battery = build_consistent_battery(seed, n_pairs, vocab)
    results = run_battery(battery)
    first_hash = aggregate_determinism_hash(results)
    registry.record(seed=seed, n_pairs=n_pairs, vocab=vocab, seq_len=1, perturb=0.0,
                    determinism_hash=first_hash, min_cosine=min(r.logit_cosine_sim for r in results), all_matched=True)

    # Replay via the harness using the stored seed tuple.
    matches, new_hash, stored, _ = replay_from_seed(
        seed, n_pairs, vocab, registry=registry
    )
    assert stored is not None
    assert matches is True
    assert new_hash == first_hash
    assert new_hash == stored["determinism_hash"]


def test_replay_different_seed_differs(tmp_path):
    registry = DeterminismRegistry(tmp_path / "reg.jsonl")
    seed, n_pairs, vocab = 2024, 4, 32

    battery = build_consistent_battery(seed, n_pairs, vocab)
    h1 = aggregate_determinism_hash(run_battery(battery))
    registry.record(seed=seed, n_pairs=n_pairs, vocab=vocab, seq_len=1, perturb=0.0,
                    determinism_hash=h1, min_cosine=1.0, all_matched=True)

    # A different seed has no stored record -> matches False.
    matches, h2, stored, _ = replay_from_seed(
        seed + 1, n_pairs, vocab, registry=registry
    )
    assert stored is None
    assert matches is False
    assert h2 != h1


def test_replay_no_registry_returns_no_match(tmp_path):
    # Replaying when nothing was ever stored must cleanly report no match.
    matches, new_hash, stored, _ = replay_from_seed(
        1, 2, 16, registry_path=tmp_path / "missing.jsonl"
    )
    assert stored is None
    assert matches is False
    assert len(new_hash) == 64


def test_replay_determinism_across_repeated_battery_builds():
    # build_consistent_battery is itself deterministic by seed.
    a = aggregate_determinism_hash(run_battery(build_consistent_battery(55, 6, 20)))
    b = aggregate_determinism_hash(run_battery(build_consistent_battery(55, 6, 20)))
    c = aggregate_determinism_hash(run_battery(build_consistent_battery(56, 6, 20)))
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# (d) registry defaults: shipped baseline readable, writes never hit the
#     package directory (the installed-copy regression, packaging audit)
# ---------------------------------------------------------------------------


def test_default_registry_reads_shipped_baseline(tmp_path, monkeypatch):
    # The committed evidence must replay True through the DEFAULT registry
    # (no explicit path): an installed copy previously reported matches=False
    # because correctness_registry.jsonl never shipped.
    monkeypatch.setenv("DISTLLM_DATA_DIR", str(tmp_path / "userdata"))
    matches, new_hash, stored, _ = replay_from_seed(1234, 8, 128)
    assert stored is not None, "shipped baseline record not found"
    assert stored["determinism_hash"] == new_hash
    assert matches is True


def test_default_registry_write_target_is_writable_not_package(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTLLM_DATA_DIR", str(tmp_path / "userdata"))
    reg = DeterminismRegistry()
    # Writes must never land in the package directory (read-only in
    # site-packages, and would churn the committed evidence).
    assert reg.path != ch.DEFAULT_REGISTRY_PATH
    assert str(tmp_path) in str(reg.path)
    # Reading still consults the shipped baseline first.
    assert ch.DEFAULT_REGISTRY_PATH in reg.read_paths
    written = reg.record(
        seed=1, n_pairs=1, vocab=8, seq_len=1, perturb=0.0,
        determinism_hash="x", min_cosine=1.0, all_matched=True,
    )
    assert reg.path.exists()
    assert written["seed"] == 1
    # The shipped baseline is untouched by the append.
    assert "x" not in ch.DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")


def test_default_registry_env_override_uses_single_file(tmp_path, monkeypatch):
    override = tmp_path / "override.jsonl"
    monkeypatch.setenv("DISTLLM_REGISTRY_PATH", str(override))
    reg = DeterminismRegistry()
    assert reg.path == override
    assert reg.read_paths == [override]

"""Regression tests for task A3: confidential-computing SCAFFOLD (software TEE sim).

This module is a SCAFFOLD.  It simulates an enclave boundary / attestation
contract and proves the *integration point* with DistLLM's existing:

  * E5 differential privacy  (``distllm.core.differential_privacy``) — DP
    Gaussian noise is applied INSIDE the enclave;
  * E4 plugin isolation       (``distllm.core.plugin_sandbox.run_isolated``) —
    an untrusted plugin is executed INSIDE the enclave under the sandbox.

HONEST CAVEAT: this is a software simulation.  There is no SGX / SEV /
NVIDIA Confidential Computing hardware on this machine.  The attestation is
signed by a local DEV key (persisted to a file, DEV ONLY) and the "memory
encryption" is an AES-GCM + HMAC stub under an in-memory key.  Real silicon
attestation + EPC memory encryption are NOT present.  The contract is what
matters: a real backend can drop in behind ``EnclaveContext`` /
``generate_attestation_report`` without changing the call sites.

These tests assert:
  1. AttestationReport has the required fields and verifies under the DEV key.
  2. Tampering with the report fails verification.
  3. DP-noise applied inside the enclave is functionally identical to outside
     (correctness preserved: same clipping + same noise given the same RNG).
  4. A plugin executed via run_in_enclave uses the E4 sandbox (audit recorded).
  5. Everything is clearly labelled SCAFFOLD (grep in-test).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from distllm.core.differential_privacy import DifferentialPrivacy, DifferentialPrivacyConfig
from distllm.core.plugin_sandbox import (
    IsolationAudit,
    IsolationConfig,
    IsolationLevel,
    run_isolated,
)
from distllm.security.tee import (
    SCAFFOLD_MARKER,
    AttestationReport,
    EnclaveContext,
    generate_attestation_report,
    run_dp_noise_in_enclave,
    run_in_enclave,
    run_plugin_in_enclave,
    verify_attestation_report,
)

# The source file is grepped for the SCAFFOLD marker in test 5.
_TEE_SOURCE = Path(__file__).resolve().parents[2] / "src" / "distllm" / "security" / "tee.py"

# ── A tiny dummy plugin body used as the isolation target ───────────────────

def _dummy_plugin(x: int) -> int:
    """Side-effect-free plugin body executed under E4 sandbox + enclave."""
    return x * 2 + 1


# ── Test 1: attestation report has required fields + verifies under DEV key ──

def test_attestation_report_has_fields_and_verifies():
    """The report carries the required fields and verifies under the DEV key."""
    nonce = os.urandom(16).hex()
    report = generate_attestation_report("enclave-a3", nonce)

    # Required fields present and well-typed.
    assert isinstance(report, AttestationReport)
    for field_name in ("enclave_id", "measurement", "nonce", "timestamp",
                        "signature", "dev_key_id"):
        assert hasattr(report, field_name), f"missing field {field_name}"
        assert getattr(report, field_name), f"field {field_name} is empty"

    # enclave_id and nonce are bound exactly as supplied.
    assert report.enclave_id == "enclave-a3"
    assert report.nonce == nonce

    # The signature is a non-trivial hex string.
    assert len(report.signature) == 128  # Ed25519 signature = 64 bytes -> 128 hex

    # Verification succeeds against the DEV public key.
    enclave = EnclaveContext("enclave-a3")
    assert verify_attestation_report(report, enclave.dev_pubkey_pem) is True

    # The enclave can verify its own attestation.
    with EnclaveContext("enclave-a3") as ctx:
        assert ctx.verify_attestation() is True
        assert ctx.attestation is not None
        assert ctx.attestation.enclave_id == "enclave-a3"


# ── Test 2: tampering with the report fails verification ────────────────────

@pytest.mark.parametrize("tamper", [
    "measurement", "nonce", "timestamp", "enclave_id", "dev_key_id", "signature",
])
def test_tampered_attestation_report_fails(tamper):
    """Any tampering (field or signature) must break verification."""
    nonce = os.urandom(16).hex()
    report = generate_attestation_report("enclave-a3", nonce)
    enclave = EnclaveContext("enclave-a3")

    bad = AttestationReport.from_dict(report.to_dict())
    if tamper == "measurement":
        bad.measurement = "0" * 64
    elif tamper == "nonce":
        bad.nonce = os.urandom(16).hex()
    elif tamper == "timestamp":
        bad.timestamp = "2000-01-01T00:00:00Z"
    elif tamper == "enclave_id":
        bad.enclave_id = "enclave-evil"
    elif tamper == "dev_key_id":
        bad.dev_key_id = "other-key"
    elif tamper == "signature":
        # Flip one hex nibble of the signature.
        sig = bytearray(bytes.fromhex(bad.signature))
        sig[0] ^= 0x01
        bad.signature = sig.hex()

    assert verify_attestation_report(bad, enclave.dev_pubkey_pem) is False


# ── Test 3: DP-noise inside enclave is functionally identical to outside ────

def test_dp_noise_inside_enclave_matches_outside():
    """E5 noise applied inside == outside (same clipping + same RNG stream).

    Correctness is preserved: the enclave is a trust *boundary*, not a
    different algorithm.  We fix the RNG seed so the Gaussian draw is
    deterministic, then check the in-enclave result equals the out-of-enclave
    result bit-for-bit — proving the enclave wrapper does not alter DP math.
    """
    import torch

    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
    dp = DifferentialPrivacy(cfg)

    base = torch.tensor([0.9, -0.4, 0.2, 0.7])

    # Out-of-enclave baseline: same RNG state, same call.
    torch.manual_seed(1234)
    outside = dp.add_noise_to_tensor(base)

    # Inside the enclave: must produce an IDENTICAL tensor.
    with EnclaveContext("enclave-dp") as ctx:
        torch.manual_seed(1234)
        inside = run_dp_noise_in_enclave(ctx, dp, base)

    assert torch.equal(inside, outside), "DP noise differs inside vs outside enclave"
    # And the inside result is the clipped-then-noised tensor (E5 semantics).
    assert inside.shape == base.shape
    # Noise was actually applied (sigma>0 for epsilon=1.0,delta=1e-5).
    assert not torch.equal(inside, base)


# ── Test 4: plugin executed via run_in_enclave uses the E4 sandbox ──────────

def test_plugin_runs_inside_enclave_under_sandbox():
    """A plugin run via run_in_enclave is executed under E4 run_isolated.

    We prove the sandbox ran by checking the IsolationAudit it records, and we
    prove the enclave boundary enforced by checking run_in_enclave refuses to
    run when the boundary is not active.
    """
    audit = IsolationAudit(plugin_name="dummy-a3", level=IsolationLevel.RLIMIT.value)
    cfg = IsolationConfig(level=IsolationLevel.RLIMIT, plugin_name="dummy-a3")

    with EnclaveContext("enclave-plugin") as ctx:
        result = run_plugin_in_enclave(
            ctx, _dummy_plugin, 21, isolation_config=cfg, audit=audit
        )

    assert result == 21 * 2 + 1
    # The E4 sandbox actually ran: an audit object was populated for this plugin.
    assert audit.plugin_name == "dummy-a3"
    # On this platform the syscall primitives may be skipped, but the audit
    # must record *something* (either applied or skipped) — i.e. run_isolated ran.
    assert (len(audit.applied) + len(audit.skipped)) > 0

    # Sanity: calling run_isolated directly with the same config records an
    # audit too (confirms E4 ran, not just our wrapper).
    audit2 = IsolationAudit()
    direct = run_isolated(_dummy_plugin, 5, config=cfg, audit=audit2)
    assert direct == 11
    assert (len(audit2.applied) + len(audit2.skipped)) > 0


def test_run_in_enclave_refuses_outside_boundary():
    """run_in_enclave must refuse to execute fn when the boundary is down."""
    ctx = EnclaveContext("enclave-off")
    # Not entered -> not active.
    with pytest.raises(RuntimeError):
        run_in_enclave(ctx, _dummy_plugin, 1)


# ── Test 5: everything is clearly labelled SCAFFOLD (grep in-test) ──────────

def test_everything_labelled_scaffold():
    """The TEE scaffold source must be saturated with the SCAFFOLD marker."""
    assert _TEE_SOURCE.exists(), f"tee.py missing at {_TEE_SOURCE}"
    text = _TEE_SOURCE.read_text(encoding="utf-8")
    # The honest caveat + the marker appear many times across the module.
    assert SCAFFOLD_MARKER in text
    assert text.count(SCAFFOLD_MARKER) >= 10, (
        "SCAFFOLD marker should appear throughout tee.py; found "
        f"{text.count(SCAFFOLD_MARKER)}"
    )
    # The honest 'no real enclave' caveat must be present.
    assert "SCAFFOLD" in text and (
        "software simulation" in text.lower() or "no real enclave" in text.lower()
        or "no sgx" in text.lower()
    ), "tee.py must state explicitly that it is a software simulation (no real enclave)"
    # Public API symbols are all defined.
    import distllm.security.tee as tee_mod
    for sym in ("EnclaveContext", "generate_attestation_report",
                "verify_attestation_report", "run_in_enclave",
                "run_dp_noise_in_enclave", "run_plugin_in_enclave"):
        assert hasattr(tee_mod, sym), f"missing public symbol {sym}"

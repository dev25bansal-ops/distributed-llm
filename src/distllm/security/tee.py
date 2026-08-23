"""SCAFFOLD — Software TEE / confidential-computing simulation for DistLLM.

=============================================================================
HONEST CAVEAT — THIS IS A SOFTWARE SIMULATION, NOT A REAL ENCLAVE.
=============================================================================
There is NO SGX / SEV / TrustZone / NVIDIA Confidential Computing hardware on
this machine.  Nothing here is cryptographically *attested by silicon*.  The
"enclave boundary", "memory encryption" and "attestation" implemented below
are SOFTWARE STUBS whose only job is to:

  (1) define the *trust boundary contract* — what code/data is considered
      "inside" the enclave vs outside, and how data crossing the boundary is
      sealed;
  (2) define the *attestation contract* — the shape of an attestation report
      and how a relying party verifies it against a (DEV, dev-only) signing
      key;
  (3) prove the *integration point* — that DistLLM's E5 differential-privacy
      noise and E4 plugin-isolation sandbox actually execute *inside* the
      enclave context, so that when a real backend (SGX SDK,
      NVIDIA CC, AWS Nitro, …) is dropped in, the call sites are already
      correct.

A real backend must replace:
  * the per-enclave key with CPU-derived (EPC) key material,
  * `seal`/`unseal` with the enclave's real memory-encryption engine,
  * `generate_attestation_report` with `sgx_report` / `sgx_quote` /
    `nvidia_get_attestation_report` (and the DEV key with the platform's
    entangled attestation key / PCCS-signed quote),
  * `verify_attestation_report` with the platform CA / PCCS verification path.

Everything in this module is explicitly marked SCAFFOLD so it is impossible to
mistake it for production confidential computing.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from loguru import logger

# Repo-local default location for the DEV attestation key.  DEV ONLY — never
# use this path in production; a real deployment obtains its attestation key
# from the enclave/platform, never from a file on disk.
_DEFAULT_DEV_KEY_PATH = (
    Path(__file__).resolve().parents[3] / ".distllm" / "tee_dev_attestation_key.pem"
)

# Marker used by the regression test that greps this file for "SCAFFOLD".
SCAFFOLD_MARKER = "SCAFFOLD"


# ─────────────────────────────────────────────────────────────────────────────
# Attestation report (SCAFFOLD contract — mirrors a real SGX quote shape)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttestationReport:
    """SCAFFOLD attestation report (analogous to an SGX `sgx_report`/`sgx_quote`).

    Fields mirror what a real remote-attestation quote carries, minus the
    silicon-signed provenance:

    Attributes:
        enclave_id: Stable identifier for the enclave instance.
        measurement: Hex SHA-256 "measurement" of the enclave code (MRENCLAVE-
            like).  For this SCAFFOLD it is a hash of the TEE module source
            plus the enclave id; a real backend would use the CPU-computed
            hash of the enclave page cache.
        nonce: Hex challenge supplied by the relying party (anti-replay).
        timestamp: ISO-8601 issuance time.
        signature: Hex Ed25519 signature over the canonical report bytes,
            produced by the DEV attestation key.  In a real backend this would
            be the platform-attested quote signature rooted at the vendor CA.
        dev_key_id: Opaque id of the DEV key that produced ``signature``
            (lets a verifier pick the right public key).  SCAFFOLD-only.
    """

    enclave_id: str
    measurement: str
    nonce: str
    timestamp: str
    signature: str
    dev_key_id: str = "dev-attestation-key"

    def _signed_bytes(self) -> bytes:
        """Canonical bytes covered by the signature (stable field order)."""
        payload = {
            "enclave_id": self.enclave_id,
            "measurement": self.measurement,
            "nonce": self.nonce,
            "timestamp": self.timestamp,
            "dev_key_id": self.dev_key_id,
        }
        return _canon(payload)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AttestationReport":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _canon(obj: dict) -> bytes:
    """Deterministic canonical JSON bytes (sorted keys, no whitespace)."""
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# DEV attestation key (SCAFFOLD — persisted to a local file, DEV ONLY)
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_create_dev_attestation_key(
    path: str | os.PathLike[str] = _DEFAULT_DEV_KEY_PATH,
) -> tuple[Ed25519PrivateKey, bytes]:
    """SCAFFOLD: load (or create + persist) the DEV Ed25519 attestation key.

    WARNING: a real attestation key is entangled with the enclave/CPU and is
    never exported to disk.  This DEV key exists only so the SCAFFOLD attestation
    contract has a stable signing identity across runs.  Do not ship it.
    """
    p = Path(path)
    if p.exists():
        priv = serialization.load_pem_private_key(p.read_bytes(), password=None)
        if not isinstance(priv, Ed25519PrivateKey):
            raise RuntimeError("SCAFFOLD: dev attestation key is not Ed25519")
        return priv, priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    p.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    logger.warning(
        "SCAFFOLD: generated DEV attestation key at %s — DEV ONLY, not a real "
        "enclave key; never use in production.",
        p,
    )
    return priv, pub_pem


def generate_attestation_report(
    enclave_id: str,
    nonce: str,
    dev_key_path: str | os.PathLike[str] = _DEFAULT_DEV_KEY_PATH,
    measurement: str | None = None,
    dev_key_id: str = "dev-attestation-key",
) -> AttestationReport:
    """SCAFFOLD: produce a signed attestation report for ``enclave_id``.

    Mirrors the contract a real backend fulfils: a relying party sends a
    ``nonce`` challenge; the enclave returns a report binding (enclave_id,
    measurement, nonce, timestamp) under its attestation key.

    Args:
        enclave_id: Identifier of the enclave instance.
        nonce: Hex (or any) challenge from the relying party.
        dev_key_path: Where the DEV attestation key lives (SCAFFOLD).
        measurement: Pre-computed hex measurement; if None, derived from the
            TEE module source + enclave_id (SCAFFOLD stand-in for MRENCLAVE).
        dev_key_id: Opaque id recorded so a verifier can pick the right key.

    Returns:
        A signed :class:`AttestationReport`.
    """
    if measurement is None:
        measurement = _compute_measurement(enclave_id)
    priv, _ = _load_or_create_dev_attestation_key(dev_key_path)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = AttestationReport(
        enclave_id=enclave_id,
        measurement=measurement,
        nonce=nonce,
        timestamp=timestamp,
        signature="",  # filled below
        dev_key_id=dev_key_id,
    )
    report.signature = priv.sign(report._signed_bytes()).hex()
    return report


def verify_attestation_report(
    report: AttestationReport,
    dev_pubkey_pem: bytes | str | Ed25519PublicKey,
) -> bool:
    """SCAFFOLD: verify an :class:`AttestationReport` against a DEV public key.

    A real backend would verify the quote against the vendor CA / PCCS instead
    of a dev key.  Returns ``True`` only if the signature is valid for the
    canonical report bytes.  Any tampering (measurement, nonce, timestamp,
    enclave_id, dev_key_id) invalidates the signature.

    Args:
        report: The report to verify.
        dev_pubkey_pem: DEV public key as PEM bytes/str, or an
            :class:`Ed25519PublicKey`.

    Returns:
        ``True`` if the signature verifies; ``False`` otherwise (never raises).
    """
    if isinstance(dev_pubkey_pem, Ed25519PublicKey):
        pub = dev_pubkey_pem
    else:
        pub = serialization.load_pem_public_key(
            dev_pubkey_pem.encode("utf-8") if isinstance(dev_pubkey_pem, str)
            else dev_pubkey_pem
        )
    if not isinstance(pub, Ed25519PublicKey):
        return False
    try:
        pub.verify(bytes.fromhex(report.signature), report._signed_bytes())
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SCAFFOLD attestation verify failed: %s", exc)
        return False


def _compute_measurement(enclave_id: str) -> str:
    """SCAFFOLD measurement (MRENCLAVE stand-in): hash of TEE module source.

    A real enclave measurement is the CPU-computed hash of the enclave page
    cache; here we hash the TEE code (so the contract binds to *this* code) and
    the enclave id.
    """
    src = Path(__file__).resolve().read_bytes()
    return hashlib.sha256(src + enclave_id.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Sealed data (SCAFFOLD "memory encryption" stub)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SealedData:
    """SCAFFOLD sealed blob: AES-GCM ciphertext + HMAC under the enclave key.

    The "memory encryption" sim: data entering the enclave is AES-256-GCM
    encrypted under a per-enclave key, and an HMAC-SHA256 over (enclave_id,
    iv, ciphertext) is recorded.  A real enclave would keep the plaintext only
    inside EPC memory; here we just keep the sealed form in the host process.
    """

    enclave_id: str
    iv: str  # hex
    ciphertext: str  # hex
    mac: str  # hex HMAC-SHA256


class EnclaveContext:
    """SCAFFOLD simulated enclave boundary (context manager).

    Entering the context:
      * marks the boundary "active",
      * derives a per-enclave "memory encryption" key (SCAFFOLD: os.urandom),
      * generates an attestation report binding the enclave measurement + a
        fresh nonce (so a relying party can verify the enclave's identity).

    Inside the context you can :meth:`seal` data (records a sealed blob + HMAC)
    and run callables via :func:`run_in_enclave`, which refuses to execute when
    the boundary is not active.

    NOTE: SCAFFOLD. No real SGX/SEV/NVIDIA-CC hardware is involved; the key is
    in host memory and the attestation is signed by a local DEV key.
    """

    def __init__(
        self,
        enclave_id: str,
        dev_key_path: str | os.PathLike[str] = _DEFAULT_DEV_KEY_PATH,
    ):
        self.enclave_id = enclave_id
        self.active = False
        # SCAFFOLD per-enclave key — a real enclave derives this from CPU/EPC.
        self._key = os.urandom(32)
        self._sealed: dict[str, SealedData] = {}
        self.dev_key_path = Path(dev_key_path)
        self._priv, self.dev_pubkey_pem = _load_or_create_dev_attestation_key(
            self.dev_key_path
        )
        self.measurement = _compute_measurement(enclave_id)
        self.attestation: AttestationReport | None = None

    def __enter__(self) -> "EnclaveContext":
        self.active = True
        # SCAFFOLD attest: bind measurement + fresh nonce under the DEV key.
        self.attestation = generate_attestation_report(
            self.enclave_id,
            nonce=os.urandom(16).hex(),
            dev_key_path=self.dev_key_path,
            measurement=self.measurement,
        )
        logger.debug("SCAFFOLD: enclave %s boundary active", self.enclave_id)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.active = False
        # SCAFFOLD: drop the in-memory key on exit (best-effort).
        self._key = b""
        self._sealed.clear()
        logger.debug("SCAFFOLD: enclave %s boundary torn down", self.enclave_id)

    # ── "memory encryption" stub ──────────────────────────────────────────

    def seal(self, name: str, data: bytes) -> SealedData:
        """SCAFFOLD: seal ``data`` under the per-enclave key (AES-GCM + HMAC)."""
        if not self.active:
            raise RuntimeError(
                "SCAFFOLD: cannot seal outside an active enclave boundary"
            )
        iv = os.urandom(12)
        ct = AESGCM(self._key).encrypt(iv, bytes(data), None)
        mac = _hmac.new(self._key, self.enclave_id.encode() + iv + ct,
                        hashlib.sha256).hexdigest()
        sealed = SealedData(
            enclave_id=self.enclave_id,
            iv=iv.hex(),
            ciphertext=ct.hex(),
            mac=mac,
        )
        self._sealed[name] = sealed
        return sealed

    def unseal(self, name: str) -> bytes:
        """SCAFFOLD: verify the HMAC and decrypt a previously sealed blob."""
        sealed = self._sealed.get(name)
        if sealed is None:
            raise KeyError(f"SCAFFOLD: no sealed data named {name!r} in enclave")
        iv = bytes.fromhex(sealed.iv)
        ct = bytes.fromhex(sealed.ciphertext)
        expected = _hmac.new(
            self._key, self.enclave_id.encode() + iv + ct, hashlib.sha256
        ).hexdigest()
        if not _hmac.compare_digest(expected, sealed.mac):
            raise RuntimeError("SCAFFOLD: sealed-data HMAC mismatch (tampered)")
        return AESGCM(self._key).decrypt(iv, ct, None)

    def verify_attestation(self) -> bool:
        """SCAFFOLD: verify this enclave's own attestation report."""
        if self.attestation is None:
            return False
        return verify_attestation_report(self.attestation, self.dev_pubkey_pem)


# ─────────────────────────────────────────────────────────────────────────────
# run_in_enclave — executes a callable INSIDE an active enclave boundary
# ─────────────────────────────────────────────────────────────────────────────

def run_in_enclave(ctx: EnclaveContext, fn: Callable[..., Any], *args: Any,
                   **kwargs: Any) -> Any:
    """SCAFFOLD: execute ``fn`` only while ``ctx`` is an active enclave.

    This is the integration seam: it asserts the trust boundary is up before
    running ``fn``, so confidential compute (E5 DP-noise, E4 plugin-exec, …)
    cannot accidentally run *outside* the enclave.

    Args:
        ctx: An active :class:`EnclaveContext`.
        fn: Callable to run inside the enclave (e.g. DP-noise or a sandboxed
            plugin).
        *args, **kwargs: Forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        RuntimeError: if the enclave boundary is not active.
    """
    if not isinstance(ctx, EnclaveContext):
        raise TypeError("SCAFFOLD: ctx must be an EnclaveContext")
    if not ctx.active:
        raise RuntimeError(
            "SCAFFOLD: enclave boundary not active — run_in_enclave refused to "
            "execute fn outside the enclave. Use `with EnclaveContext(...) as "
            "ctx:` first."
        )
    return fn(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Integration wrappers — E5 DP-noise and E4 plugin-exec INSIDE the enclave
# ─────────────────────────────────────────────────────────────────────────────
# These ONLY CALL the existing E5/E4 code from inside run_in_enclave.  They do
# NOT modify differential_privacy.py or plugin_sandbox.py internals.

def run_dp_noise_in_enclave(
    ctx: EnclaveContext, dp, tensor: Any, *args: Any, **kwargs: Any
) -> Any:
    """SCAFFOLD integration: apply E5 DP Gaussian noise INSIDE the enclave.

    Calls ``dp.add_noise_to_tensor(tensor, ...)`` (the existing E5 mechanism)
    under :func:`run_in_enclave`, so the privacy-preserving noise is applied
    within the trust boundary.  The DP object is whatever the caller already
    uses (e.g. ``DifferentialPrivacy`` / ``PrivacyAccountant``-backed helper);
    we only invoke its public ``add_noise_to_tensor``.
    """
    return run_in_enclave(ctx, dp.add_noise_to_tensor, tensor, *args, **kwargs)


def run_plugin_in_enclave(
    ctx: EnclaveContext,
    plugin_fn: Callable[..., Any],
    *args: Any,
    isolation_config: Any = None,
    audit: Any = None,
    **kwargs: Any,
) -> Any:
    """SCAFFOLD integration: run a plugin via E4 sandbox INSIDE the enclave.

    Calls ``plugin_sandbox.run_isolated(plugin_fn, ...)`` (the existing E4
    OS-gated isolation) under :func:`run_in_enclave`, proving the untrusted
    plugin executes both *inside the enclave boundary* AND *under the sandbox*.
    """
    from distllm.core.plugin_sandbox import run_isolated  # E4 — imported, not modified

    def _run() -> Any:
        return run_isolated(
            plugin_fn, *args, config=isolation_config, audit=audit, **kwargs
        )

    return run_in_enclave(ctx, _run)


__all__ = [
    "SCAFFOLD_MARKER",
    "AttestationReport",
    "SealedData",
    "EnclaveContext",
    "generate_attestation_report",
    "verify_attestation_report",
    "run_in_enclave",
    "run_dp_noise_in_enclave",
    "run_plugin_in_enclave",
    "_load_or_create_dev_attestation_key",
    "_DEFAULT_DEV_KEY_PATH",
]

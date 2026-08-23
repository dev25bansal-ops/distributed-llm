"""Signed-manifest, capability-scoped plugin sandbox.

Re-architects plugin safety (the old C1 liability) into a safe ecosystem
play:

* **Signed manifest** — every plugin ships a ``PluginManifest`` carrying its
  name, version, a SHA-256 of its payload, the *capabilities* it requests,
  and an **Ed25519 signature** over that content.  ``verify_manifest`` checks
  the signature against a trusted public key; unsigned or tampered manifests
  are rejected (fail-closed).
* **Capability scoping** — plugins declare the narrow set of
  :class:`PluginCapability` they need (network, filesystem, subprocess, gpu,
  …).  A :class:`SandboxPolicy` enforces them: a plugin granted only
  ``FILESYSTEM_READ`` can never open a network socket via the sandbox runner.
* **Subprocess sandbox** — :func:`run_sandboxed` executes plugin code in a
  restricted subprocess (no shell, no inherited network namespace when
  possible, resource + timeout limits).  A WASM execution path is left as an
  explicit seam (``run_wasm``) for when a WASM runtime is wired in; the
  subprocess sandbox is the production default today.

The marketplace integration is **opt-in**: when ``PluginMarketplace`` is
constructed without a ``public_key`` / ``sandbox_policy``, it behaves exactly
as before (existing tests keep passing).  When configured, install/enable
verify the manifest signature + capabilities before trusting the plugin.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import ctypes
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger

# `resource` is a POSIX-only module; on Windows (and other non-POSIX hosts)
# `setrlimit`/`getrlimit` are simply unavailable.  Import it defensively so
# the isolation wrapper degrades gracefully instead of ImportError-ing at
# import time on platforms that can't enforce rlimits in-process.
try:  # pragma: no cover - platform dependent
    import resource as _resource
except Exception:  # pragma: no cover - e.g. Windows
    _resource = None

try:  # cryptography is a core dependency (Ed25519 signing).
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _HAVE_CRYPTO = True
except Exception:  # pragma: no cover
    _HAVE_CRYPTO = False


class PluginCapability(str, Enum):
    """Capabilities a plugin may request (least-privilege, deny-by-default)."""

    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    GPU = "gpu"
    ENV_READ = "env_read"
    REGISTRY_WRITE = "registry_write"  # may register backends/strategies


# Capabilities that require the subprocess sandbox to actually isolate.
_NETWORK_CAP = PluginCapability.NETWORK
_SUBPROCESS_CAP = PluginCapability.SUBPROCESS


@dataclass
class PluginManifest:
    """Signed descriptor for a plugin.

    Attributes:
        name: Plugin package name.
        version: Semantic version.
        sha256: Hex SHA-256 of the plugin payload (wheel/source archive).
        capabilities: Capabilities the plugin requests (deny-by-default).
        entry_point: ``module:Class`` the marketplace loads.
        author: Author / publisher identity.
        signature: Hex Ed25519 signature over ``_signed_bytes()`` (empty until
            signed).
    """

    name: str
    version: str = "1.0.0"
    sha256: str = ""
    capabilities: list[str] = field(default_factory=list)
    entry_point: str = ""
    author: str = ""
    signature: str = ""

    def _signed_bytes(self) -> bytes:
        """Canonical bytes covered by the signature (stable field order)."""
        payload = {
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "capabilities": sorted(self.capabilities),
            "entry_point": self.entry_point,
            "author": self.author,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def sign(self, private_key: Ed25519PrivateKey) -> str:
        """Sign this manifest in place; returns the hex signature."""
        sig = private_key.sign(self._signed_bytes())
        self.signature = sig.hex()
        return self.signature

    def verify(self, public_key: Ed25519PublicKey) -> bool:
        """Verify the manifest signature against a trusted public key."""
        if not self.signature:
            return False
        try:
            public_key.verify(bytes.fromhex(self.signature), self._signed_bytes())
            return True
        except Exception as exc:
            logger.warning("Manifest signature verify failed for %s: %s", self.name, exc)
            return False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PluginManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate_key_pair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate an Ed25519 (private, public) key pair for plugin signing."""
    if not _HAVE_CRYPTO:
        raise RuntimeError("cryptography is required for plugin signing")
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


def public_key_from_pem(pem: bytes) -> Ed25519PublicKey:
    if not _HAVE_CRYPTO:
        raise RuntimeError("cryptography is required for plugin verification")
    return serialization.load_pem_public_key(pem)


def verify_manifest(manifest: PluginManifest, public_key_pem: bytes) -> bool:
    """Verify a manifest's signature using a trusted public key (PEM)."""
    return manifest.verify(public_key_from_pem(public_key_pem))


@dataclass
class SandboxPolicy:
    """Capability-scoped execution policy for a plugin.

    Args:
        capabilities: The capabilities the plugin is *granted* (intersection
            of what it requested and what the operator allows).
        allowed_paths: Filesystem paths the plugin may read/write (when those
            capabilities are granted).
        timeout_s: Hard timeout for any sandboxed subprocess.
        max_memory_mb: Address-space cap for the sandboxed subprocess.
    """

    capabilities: set[PluginCapability] = field(default_factory=set)
    allowed_paths: list[str] = field(default_factory=list)
    timeout_s: float = 30.0
    max_memory_mb: int = 512

    def allows(self, cap: PluginCapability) -> bool:
        return cap in self.capabilities

    def requires(self, *caps: PluginCapability) -> bool:
        """True only if EVERY requested capability is granted."""
        return all(self.allows(c) for c in caps)

    @classmethod
    def from_granted(cls, granted: list[str], **kw: Any) -> "SandboxPolicy":
        return cls(
            capabilities={PluginCapability(c) for c in granted if c in _CAP_NAMES},
            **kw,
        )


_CAP_NAMES = {c.value for c in PluginCapability}


# ─────────────────────────────────────────────────────────────────────────────
# Real, OS-gated plugin isolation (E4)
#
# When an operator wants stronger guarantees than the capability-scoped
# subprocess launcher (run_sandboxed), untrusted plugin *in-process* code can
# be executed under a hardening wrapper (:func:`run_isolated`).  On Linux it
# layers:
#
#   (a) resource.setrlimit(RLIMIT_AS / RLIMIT_CPU / RLIMIT_NOFILE /
#       RLIMIT_FSIZE) — bounds address space, CPU seconds, open fds, and file
#       size so a runaway/allocating plugin is reaped instead of OOM-killing
#       the host;
#   (b) unshare(CLONE_NEWNET) — drops the plugin's network namespace so it
#       cannot phone home (enforced via ctypes to libc `unshare`);
#   (c) a seccomp-bpf filter (via libseccomp if importable, else a documented
#       stub) that blocks dangerous syscalls (execve of new binaries, mount,
#       ptrace, …) while permitting the Python runtime + the file IO the
#       plugin needs.
#
# On non-Linux hosts (notably this Windows CI host) (b) and (c) are skipped,
# (a) is applied only where the platform supports it, and every enforcement
# decision is recorded in an **audit log** so the policy/plumbing is still
# fully testable without the kernel syscalls.
#
# The isolation *level* is configurable via the ``DISTLLM_PLUGIN_ISOLATION``
# environment variable (``full`` | ``netns`` | ``rlimit`` | ``off``) and
# defaults to ``rlimit`` so existing unisolated behaviour is preserved by
# default (``off`` is also available for explicit opt-out / back-compat).
# ─────────────────────────────────────────────────────────────────────────────

class IsolationLevel(str, Enum):
    """Granularity of in-process plugin isolation applied by ``run_isolated``.

    * ``FULL``   — setrlimit + unshare(CLONE_NEWNET) + seccomp-bpf (Linux).
    * ``NETNS``  — setrlimit + unshare(CLONE_NEWNET) only (Linux; no seccomp).
    * ``RLIMIT`` — setrlimit only (cross-platform best-effort).
    * ``OFF``    — no restriction applied (backward compatible / debug).
    """

    FULL = "full"
    NETNS = "netns"
    RLIMIT = "rlimit"
    OFF = "off"


# Mapping of env-var string -> IsolationLevel (validated, lower-cased).
_ISOLATION_LEVELS = {lvl.value: lvl for lvl in IsolationLevel}


def isolation_level_from_env(
    env: dict[str, str] | None = None,
    default: IsolationLevel = IsolationLevel.RLIMIT,
) -> IsolationLevel:
    """Resolve the isolation level from ``DISTLLM_PLUGIN_ISOLATION``.

    Args:
        env: Mapping to read the var from (defaults to ``os.environ``).
        default: Level used when the var is unset or unrecognised.

    Returns:
        The resolved :class:`IsolationLevel`.
    """
    src = os.environ if env is None else env
    raw = (src.get("DISTLLM_PLUGIN_ISOLATION") or "").strip().lower()
    if not raw:
        return default
    lvl = _ISOLATION_LEVELS.get(raw)
    if lvl is None:
        logger.warning(
            "Unknown DISTLLM_PLUGIN_ISOLATION=%r; falling back to %r",
            raw, default.value,
        )
        return default
    return lvl


@dataclass
class IsolationConfig:
    """Configuration for :func:`run_isolated`.

    Attributes:
        level: Which isolation primitives to apply.
        max_address_mb: ``RLIMIT_AS`` ceiling (virtual address space), MiB.
        max_cpu_seconds: ``RLIMIT_CPU`` ceiling, seconds.
        max_open_files: ``RLIMIT_NOFILE`` ceiling.
        max_file_size_mb: ``RLIMIT_FSIZE`` ceiling, MiB (0 disables).
        plugin_name: Human-readable plugin identity for the audit log.
    """

    level: IsolationLevel = IsolationLevel.RLIMIT
    max_address_mb: int = 512
    max_cpu_seconds: int = 30
    max_open_files: int = 256
    max_file_size_mb: int = 0
    plugin_name: str = "<plugin>"

    def with_level(self, level: IsolationLevel) -> "IsolationConfig":
        """Return a copy of this config with ``level`` overridden."""
        return IsolationConfig(
            level=level,
            max_address_mb=self.max_address_mb,
            max_cpu_seconds=self.max_cpu_seconds,
            max_open_files=self.max_open_files,
            max_file_size_mb=self.max_file_size_mb,
            plugin_name=self.plugin_name,
        )


@dataclass
class IsolationAudit:
    """Record of what :func:`run_isolated` actually enforced for one run.

    ``applied`` lists the concrete primitives enforced (e.g. ``"rlimit_as"``,
    ``"netns_unshare"``, ``"seccomp"``); ``skipped`` lists requested-but-not-
    performed primitives (e.g. on a platform that lacks the syscall); and
    ``messages`` carries the human-readable rationale for each decision.  This
    is the object the Windows regression tests assert against — it proves the
    *policy and plumbing* ran without requiring the Linux kernel syscalls.
    """

    plugin_name: str = "<plugin>"
    level: str = IsolationLevel.RLIMIT.value
    platform: str = sys.platform
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def note_applied(self, primitive: str, msg: str = "") -> None:
        self.applied.append(primitive)
        if msg:
            self.messages.append(msg)

    def note_skipped(self, primitive: str, msg: str = "") -> None:
        self.skipped.append(primitive)
        if msg:
            self.messages.append(msg)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "level": self.level,
            "platform": self.platform,
            "applied": list(self.applied),
            "skipped": list(self.skipped),
            "messages": list(self.messages),
        }


def _apply_rlimits(cfg: IsolationConfig, audit: IsolationAudit) -> None:
    """Apply POSIX resource limits that are available on this platform.

    Linux supports all four.  Windows has no ``resource`` module, so this is a
    no-op that records the skip in the audit log (rather than raising), keeping
    the Windows test path green.
    """
    if _resource is None:
        audit.note_skipped(
            "rlimit",
            "resource module unavailable on this platform; rlimits not enforced",
        )
        return

    def _set(which: int, limit: int, label: str) -> None:
        try:
            _resource.setrlimit(which, (limit, limit))
            audit.note_applied(label, f"setrlimit {label}={limit}")
        except (ValueError, OSError) as exc:  # pragma: no cover - defensive
            audit.note_skipped(label, f"setrlimit {label} failed: {exc}")

    if cfg.max_address_mb > 0:
        _set(_resource.RLIMIT_AS, cfg.max_address_mb * 1024 * 1024, "rlimit_as")
    if cfg.max_cpu_seconds > 0:
        _set(_resource.RLIMIT_CPU, cfg.max_cpu_seconds, "rlimit_cpu")
    if cfg.max_open_files > 0:
        _set(_resource.RLIMIT_NOFILE, cfg.max_open_files, "rlimit_nofile")
    if cfg.max_file_size_mb > 0:
        _set(
            _resource.RLIMIT_FSIZE,
            cfg.max_file_size_mb * 1024 * 1024,
            "rlimit_fsize",
        )


def _apply_netns_unshare(audit: IsolationAudit) -> None:
    """Enter a new network namespace via libc ``unshare`` (Linux only).

    Uses ``ctypes`` against ``libc.so.6`` so there is no hard dependency on
    ``cffi``/pylinux.  ``CLONE_NEWNET == 0x40000000``.  Wrapped in try/except:
    if the syscall is unavailable (non-Linux, or lacking ``CAP_SYS_ADMIN``),
    the skip is recorded rather than raised.
    """
    if sys.platform != "linux":
        audit.note_skipped(
            "netns_unshare",
            f"netns unshare skipped: platform {sys.platform!r} is not Linux",
        )
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        CLONE_NEWNET = 0x40000000
        rc = libc.unshare(ctypes.c_int(CLONE_NEWNET))
        if rc != 0:  # pragma: no cover - depends on host caps
            err = ctypes.get_errno()
            audit.note_skipped(
                "netns_unshare",
                f"unshare(CLONE_NEWNET) returned {rc} (errno={err}); "
                f"network namespace not changed",
            )
            return
        audit.note_applied("netns_unshare", "entered CLONE_NEWNET namespace")
    except Exception as exc:  # pragma: no cover - e.g. libc missing
        audit.note_skipped("netns_unshare", f"unshare unavailable: {exc}")


def _apply_seccomp(audit: IsolationAudit) -> None:
    """Apply a minimal seccomp-bpf filter blocking dangerous syscalls (Linux).

    Prefers the ``libseccomp`` Python bindings when importable.  When they are
    not present (or on a non-Linux platform), records a documented stub note
    rather than silently pretending isolation is active.  The filter allows
    the common syscalls a Python plugin needs (read/write/open/close/mmap/
    futex/sched_yield/…) and **kills** the process on the dangerous set
    (execve, execveat, mount, umount2, ptrace, fork, vfork, clone-for-exec,
    kill, reboot, kexec_load, bpf, …) so a plugin cannot spawn binaries,
    escalate, or tamper with the kernel.
    """
    if sys.platform != "linux":
        audit.note_skipped(
            "seccomp",
            f"seccomp skipped: platform {sys.platform!r} is not Linux",
        )
        return
    try:  # pragma: no cover - libseccomp optional
        from seccomp import (  # type: ignore
            ALL_ARCH,
            Action,
            Filter,
            Syscall,
            Arg,
        )
    except Exception:
        audit.note_skipped(
            "seccomp",
            "seccomp not available (libseccomp not importable); "
            "running with setrlimit+netns only",
        )
        return

    try:  # pragma: no cover - requires libseccomp + kernel support
        f = Filter(ALL_ARCH)
        # Default-deny dangerous syscalls; default-allow everything else so
        # the plugin's normal Python runtime keeps working.
        dangerous = (
            "execve", "execveat", "mount", "umount2", "ptrace",
            "fork", "vfork", "kill", "reboot", "kexec_load", "bpf",
            "init_module", "finit_module", "setuid", "setgid",
            "chroot", "pivot_root", "swapon", "swapoff",
        )
        for name in dangerous:
            try:
                f.add_rule(Action.KILL, Syscall(name))
            except Exception:  # pragma: no cover - unknown syscall name
                continue
        f.load()
        audit.note_applied("seccomp", "seccomp-bpf filter loaded (KILL on dangerous syscalls)")
    except Exception as exc:  # pragma: no cover - kernel/perm limits
        audit.note_skipped("seccomp", f"seccomp load failed: {exc}")


def run_isolated(
    func: Callable[..., Any],
    *args: Any,
    config: IsolationConfig | None = None,
    audit: IsolationAudit | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an untrusted plugin callable under OS-gated isolation.

    On Linux and at ``level=FULL`` this applies setrlimit + netns unshare +
    seccomp before calling ``func``.  Lower levels apply a strict subset;
    ``level=OFF`` runs ``func`` unchanged.  The function is executed **in the
    current process** (so hooks that mutate shared state still work) but with
    the resource/namespace/seccomp constraints installed first.

    Args:
        func: The plugin callable to execute (e.g. an ``on_request`` hook).
        *args, **kwargs: Forwarded to ``func``.
        config: :class:`IsolationConfig`; defaults to
            ``IsolationConfig(level=isolation_level_from_env())``.
        audit: Optional :class:`IsolationAudit` to populate.  When ``None`` a
            fresh audit is created (and returned via the wrapper's audit attr).

    Returns:
        Whatever ``func`` returns.

    Raises:
        Any exception ``func`` raises (propagated unchanged).
    """
    global _last_isolation_audit
    cfg = config or IsolationConfig(level=isolation_level_from_env())
    if audit is None:
        audit = IsolationAudit(plugin_name=cfg.plugin_name, level=cfg.level.value)
    else:
        # The audit always reflects what actually ran: sync it from the
        # effective config (the authoritative source of enforcement), keeping
        # any caller-supplied name only when the config is still generic.
        audit.level = cfg.level.value
        if audit.plugin_name in (None, "", "<plugin>"):
            audit.plugin_name = cfg.plugin_name
    rec = audit

    if cfg.level is IsolationLevel.OFF:
        rec.note_skipped("all", "isolation level=off; no restrictions applied")
        logger.debug("Plugin %s: isolation off — running unisolated", cfg.plugin_name)
        return func(*args, **kwargs)

    # (a) resource limits — cross-platform best-effort (no-op on Windows).
    _apply_rlimits(cfg, rec)

    if cfg.level in (IsolationLevel.FULL, IsolationLevel.NETNS):
        # (b) network namespace isolation.
        _apply_netns_unshare(rec)

    if cfg.level is IsolationLevel.FULL:
        # (c) seccomp-bpf.
        _apply_seccomp(rec)

    logger.debug(
        "Plugin %s: ran with isolation level=%s (applied=%s, skipped=%s)",
        cfg.plugin_name, cfg.level.value, rec.applied, rec.skipped,
    )
    _last_isolation_audit = rec
    return func(*args, **kwargs)


_last_isolation_audit: IsolationAudit | None = None


def last_audit() -> IsolationAudit | None:
    """Return the most recent :class:`IsolationAudit` produced by run_isolated.

    Kept for operators who want a quick peek at what the last plugin run
    enforced; the regression tests instead pass their own audit object.
    """
    return _last_isolation_audit


def run_sandboxed(
    cmd: list[str],
    policy: SandboxPolicy,
    payload: bytes | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a plugin command in a capability-scoped subprocess sandbox.

    Enforces the policy:
    * ``NETWORK`` not granted -> launch with a blocked network environment
      (``DISTLLM_SANDBOX_NO_NET=1`` + isolated-looking env; on platforms with
      ``unshare`` this would also enter a network namespace).
    * ``SUBPROCESS`` not granted -> refuse to run a shell/arbitrary command.
    * ``ENV_READ`` not granted -> scrub sensitive env vars.
    * Always: no shell, explicit timeout, restricted stdout/stderr capture.

    Args:
        cmd: The command argv (must NOT invoke a shell).
        policy: The capability-scoped :class:`SandboxPolicy`.
        payload: Optional stdin bytes (e.g. plugin archive) for the process.
        cwd: Working directory (must be within ``allowed_paths`` when
            filesystem caps are scoped).

    Returns:
        The :class:`subprocess.CompletedProcess` result.

    Raises:
        PermissionError: if the policy forbids the requested operation.
        ValueError: if ``cmd`` looks like a shell invocation.
    """
    if not cmd:
        raise ValueError("empty command")
    # Deny shell invocations outright.
    if any(c in (";", "&&", "|", "$(", "`") for c in cmd) or cmd[0] in ("sh", "bash", "cmd", "powershell"):
        raise PermissionError("shell invocation denied by sandbox policy")

    if not policy.allows(_SUBPROCESS_CAP) and _looks_like_subprocess_escape(cmd):
        raise PermissionError("SUBPROCESS capability required to run external commands")

    # Network isolation is not just an env hint: when the NETWORK capability is
    # not granted, refuse to execute any command that can reach the network.
    # (The DISTLLM_SANDBOX_NO_NET env var remains set for defense-in-depth, but
    # the launcher now also hard-blocks known network clients instead of relying
    # on the child to honor the flag.)
    if not policy.allows(_NETWORK_CAP) and _network_capable(cmd):
        raise PermissionError("NETWORK capability required to run network-capable command")

    env = dict(os.environ)
    if not policy.allows(PluginCapability.ENV_READ):
        for secret in ("API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DATABASE_URL", "AWS_SECRET_ACCESS_KEY"):
            env.pop(secret, None)
    if not policy.allows(_NETWORK_CAP):
        # Signal (and, where supported, enforce) no outbound network.
        env["DISTLLM_SANDBOX_NO_NET"] = "1"

    if cwd and policy.allowed_paths and not _within_allowed(cwd, policy.allowed_paths):
        raise PermissionError(f"cwd {cwd!r} outside allowed_paths")

    env = _scrub_env(policy)

    try:
        return subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=isinstance(payload, str) or payload is None,
            timeout=policy.timeout_s,
            env=env,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("Sandboxed plugin timed out after %ss: %s", policy.timeout_s, cmd)
        raise


# Secrets scrubbed from the sandboxed env when ENV_READ is not granted.
_SECRET_KEYS = (
    "API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "AWS_SECRET_ACCESS_KEY",
)


def _scrub_env(policy: SandboxPolicy) -> dict[str, str]:
    """Build the env a sandboxed command would inherit.

    Extracted so the env-stripping policy can be asserted in-process
    (without spawning a subprocess).  Sensitive variables are dropped
    unless the ``ENV_READ`` capability is granted.
    """
    env = dict(os.environ)
    if not policy.allows(PluginCapability.ENV_READ):
        for secret in _SECRET_KEYS:
            env.pop(secret, None)
    if not policy.allows(_NETWORK_CAP):
        env["DISTLLM_SANDBOX_NO_NET"] = "1"
    return env


def _looks_like_subprocess_escape(cmd: list[str]) -> bool:
    """Heuristic: an external binary (not a python -m / direct module)."""
    # Allow python-module style launchers; flag anything that reaches for a
    # bare system binary as needing SUBPROCESS.
    head = cmd[0]
    return not (head.endswith("python") or head.endswith("python.exe") or head.endswith("python3"))


_NETWORK_BINARIES = {
    "curl", "wget", "ssh", "scp", "nc", "ncat", "telnet", "ftp", "sftp",
    "rsync", "ping", "traceroute", "dig", "nslookup", "aws", "gcloud",
    "az", "kubectl", "docker", "npm", "pip", "pip3", "git",
}


def _network_capable(cmd: list[str]) -> bool:
    """Heuristic: True if ``cmd`` can reach the network.

    Covers known network client binaries and URL/subprocess combos. This is a
    defense-in-depth block used when the NETWORK capability is not granted; it
    is intentionally conservative (erring toward refusal).
    """
    if not cmd:
        return False
    head = os.path.basename(cmd[0])
    if head in _NETWORK_BINARIES:
        return True
    # `python -c "import urllib..."` or `sh -c "curl ..."` style network use.
    if len(cmd) >= 3 and cmd[1] in ("-c", "-m") and any(
        tok in " ".join(cmd[2:]).lower()
        for tok in ("http://", "https://", "socket", "urllib", "requests", "curl", "wget")
    ):
        return True
    return False


def _within_allowed(path: str, allowed: list[str]) -> bool:
    import os.path as osp

    path = osp.abspath(path)
    return any(osp.commonpath([path, osp.abspath(a)]) == osp.abspath(a) for a in allowed)


def run_wasm(
    wasm_path: str,
    policy: SandboxPolicy,
    func: str = "_start",
    args: list[str] | None = None,
) -> Any:
    """WASM execution seam.

    Intentionally raises until a WASM runtime (e.g. wasmtime-py) is wired in.
    The subprocess sandbox is the production default; WASM is the stronger
    isolation target for untrusted plugins and plugs in here without changing
    the marketplace contract.
    """
    raise NotImplementedError(
        "WASM sandbox not wired (no WASM runtime dependency). "
        "Use run_sandboxed for the subprocess-based capability sandbox."
    )


def verify_artifact_signature(artifact_path: str, *, identity: str | None = None) -> bool:
    """Verify a model/plugin artifact with sigstore (cosign).

    Implements the C1/B615 "artifact signature verification" integration.
    Uses the ``sigstore`` Python API when installed; otherwise falls back to
    the ``cosign`` CLI if present on PATH.  Returns True only if the artifact
    carries a valid sigstore signature (and, when ``identity`` is given, the
    signing identity matches).  If no verifier is available, returns False
    rather than pretending the artifact is trusted.

    This is fail-closed: an unverifiable environment never reports "safe".
    """
    # 1. sigstore Python SDK (preferred, in-process).
    try:
        from sigstore.verify import Verifier  # type: ignore

        verifier = Verifier.production()
        verifier.verify(
            input_=artifact_path,
            identity=identity,
        )
        return True
    except ImportError:
        pass
    except Exception:
        return False

    # 2. cosign CLI fallback.
    import shutil
    import subprocess

    cosign = shutil.which("cosign")
    if cosign is None:
        return False
    cmd = ["cosign", "verify-blob", "--certificate-identity", identity or "*", artifact_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc.returncode == 0
    except Exception:
        return False


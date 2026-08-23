"""Regression tests for HIGH fix E4: real plugin isolation (OS-gated).

Adds a hardened, OS-gated execution wrapper (:func:`run_isolated`) on top of
the capability-scoped sandbox:

  * On Linux it layers ``resource.setrlimit`` (RLIMIT_AS / RLIMIT_CPU /
    RLIMIT_NOFILE / RLIMIT_FSIZE), ``unshare(CLONE_NEWNET)`` (network
    namespace drop via ctypes to libc), and a ``seccomp-bpf`` filter
    (libseccomp if importable, else a documented stub).
  * On non-Linux hosts (this Windows CI host) the kernel syscall paths are
    skipped but the wrapper still applies ``setrlimit`` where available and
    records an **audit log** of every enforcement decision.  The isolation
    LEVEL is configurable via ``DISTLLM_PLUGIN_ISOLATION`` (``full`` /
    ``netns`` / ``rlimit`` / ``off``).

These tests exercise the *policy and plumbing*, not the Linux kernel syscalls,
so they run green on Windows.  They assert:

  1. The wrapper applies setrlimit bounds and records an isolation audit entry
     without raising (on Windows rlimit is recorded as skipped — the audit is
     what matters).
  2. The isolation level is configurable via the ``DISTLLM_PLUGIN_ISOLATION``
     environment variable.
  3. When ``level='off'`` no restriction is applied (backward compatible).
"""

from __future__ import annotations

import os

import pytest

from distllm.core.plugin_sandbox import (
    IsolationAudit,
    IsolationConfig,
    IsolationLevel,
    isolation_level_from_env,
    run_isolated,
)
from distllm.core.plugin_system import PluginBase, PluginMetadata, PluginSystem


# ── A tiny dummy "plugin" callable the wrapper executes ──────────────────────

def _dummy_plugin(x: int) -> int:
    """A harmless, side-effect-free plugin body used as the isolation target."""
    return x * 2 + 1


class _DummyPlugin(PluginBase):
    """A tiny plugin whose hooks we dispatch under isolation."""

    def name(self) -> str:
        return "dummy-e4"

    def on_request(self, context):
        context.setdefault("seen", []).append(self.name())
        return {"touched": True}


# ── Test 1: wrapper runs the plugin and records an isolation audit ──────────

def test_run_isolated_applies_and_audits_on_windows():
    """On Windows the syscall paths are skipped, but the audit must be recorded
    and the plugin must run and return its result (no raise)."""
    audit = IsolationAudit()
    cfg = IsolationConfig(
        level=IsolationLevel.RLIMIT,
        plugin_name="dummy-e4",
        max_address_mb=256,
        max_cpu_seconds=10,
        max_open_files=128,
    )

    result = run_isolated(_dummy_plugin, 21, config=cfg, audit=audit)

    # The plugin executes normally and returns its value.
    assert result == 21 * 2 + 1

    # An isolation audit entry was produced for this run.
    assert audit.plugin_name == "dummy-e4"
    assert audit.level == "rlimit"
    # Either rlimit was applied (Linux) or explicitly skipped (Windows) — both
    # are valid; what we assert is that the policy/plumbing ran and decided.
    assert "rlimit" in audit.skipped or any(
        a.startswith("rlimit_") for a in audit.applied
    )
    # Network namespace + seccomp are not part of the `rlimit` level, so they
    # must NOT be in the applied set.
    assert "netns_unshare" not in audit.applied
    assert "seccomp" not in audit.applied


# ── Test 2: isolation level is configurable via env ─────────────────────────

@pytest.mark.parametrize(
    "env_val,expected",
    [
        ("full", IsolationLevel.FULL),
        ("netns", IsolationLevel.NETNS),
        ("rlimit", IsolationLevel.RLIMIT),
        ("off", IsolationLevel.OFF),
        ("", IsolationLevel.RLIMIT),          # default when unset
        ("bogus", IsolationLevel.RLIMIT),     # unknown -> default, no raise
    ],
)
def test_isolation_level_from_env(monkeypatch, env_val, expected):
    monkeypatch.delenv("DISTLLM_PLUGIN_ISOLATION", raising=False)
    if env_val:
        monkeypatch.setenv("DISTLLM_PLUGIN_ISOLATION", env_val)
    assert isolation_level_from_env() == expected


def test_run_isolated_respects_env_level(monkeypatch):
    """run_isolated reads DISTLLM_PLUGIN_ISOLATION when no config is passed."""
    monkeypatch.setenv("DISTLLM_PLUGIN_ISOLATION", "netns")
    audit = IsolationAudit()
    # No config -> resolves level from env ("netns").
    result = run_isolated(_dummy_plugin, 5, audit=audit)
    assert result == 11
    assert audit.level == "netns"
    # netns is requested at the "netns" level -> must appear in applied OR
    # skipped (on Windows it is skipped with a clear rationale).
    assert "netns_unshare" in audit.applied or "netns_unshare" in audit.skipped
    # seccomp is NOT part of "netns" -> never applied.
    assert "seccomp" not in audit.applied


# ── Test 3: level='off' applies no restriction ─────────────────────────────

def test_run_isolated_off_applies_no_restriction():
    audit = IsolationAudit()
    cfg = IsolationConfig(level=IsolationLevel.OFF, plugin_name="dummy-e4")
    result = run_isolated(_dummy_plugin, 7, config=cfg, audit=audit)

    assert result == 15
    # At level=off nothing is applied and the audit notes the opt-out.
    assert audit.applied == []
    assert "all" in audit.skipped


def test_plugin_dispatch_under_off_isolation_passthrough():
    """Backward compat: a PluginSystem with isolation off (default) dispatches
    hooks unchanged and still returns the plugin's result."""
    sys_off = PluginSystem()
    # Default config -> isolation_config is None -> passthrough (no wrapper).
    from distllm.core.plugin_system import PluginInstance, PluginState
    meta = _DummyPlugin().metadata()
    inst = PluginInstance(_DummyPlugin, meta)
    inst.instance = _DummyPlugin()
    inst.state = PluginState.STARTED
    sys_off._plugins["dummy-e4"] = inst

    out = sys_off.dispatch_on_request({"prompt": "hi"})
    assert out["touched"] is True
    assert "dummy-e4" in out["seen"]


def test_plugin_dispatch_under_rlimit_isolation_runs():
    """When isolation is opted in (rlimit), dispatch still executes the hook
    and the isolation audit reflects the configured level."""
    sys_iso = PluginSystem(config={"isolation_config": IsolationConfig(
        level=IsolationLevel.RLIMIT, plugin_name="dummy-e4",
    )})
    from distllm.core.plugin_system import PluginInstance, PluginState
    meta = _DummyPlugin().metadata()
    inst = PluginInstance(_DummyPlugin, meta)
    inst.instance = _DummyPlugin()
    inst.state = PluginState.STARTED
    sys_iso._plugins["dummy-e4"] = inst

    out = sys_iso.dispatch_on_request({"prompt": "hi"})
    assert out["touched"] is True
    # The audit stamp recorded the real plugin identity.
    from distllm.core.plugin_sandbox import last_audit
    assert last_audit() is not None
    assert last_audit().plugin_name == "dummy-e4"
    assert last_audit().level == "rlimit"

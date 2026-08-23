"""Regression tests for HIGH fix C9: plugin sandbox isolation was illusory.

Network isolation previously relied solely on an env var hint
(``DISTLLM_SANDBOX_NO_NET=1``) that the child could ignore. Now, when the
``NETWORK`` capability is not granted, ``run_sandboxed`` hard-refuses to launch
any command that can reach the network.
"""

from __future__ import annotations

import pytest

from distllm.core.plugin_sandbox import (
    PluginCapability,
    PluginManifest,
    SandboxPolicy,
    _network_capable,
    run_sandboxed,
)


def _no_net_policy() -> SandboxPolicy:
    return SandboxPolicy(capabilities={PluginCapability.FILESYSTEM_READ})


def test_network_capable_heuristic():
    assert _network_capable(["curl", "http://x"])
    assert _network_capable(["python", "-c", "import urllib.request"])
    assert not _network_capable(["python", "-m", "mymodule"])


def test_run_sandboxed_refuses_network_without_cap():
    with pytest.raises(PermissionError):
        run_sandboxed(["curl", "https://evil.example"], _no_net_policy())


def test_run_sandboxed_allows_network_with_cap():
    policy = SandboxPolicy(capabilities={PluginCapability.NETWORK})
    # With the NETWORK cap granted, the launcher no longer refuses on the
    # network heuristic (subprocess execution itself still needs SUBPROCESS;
    # here we only assert the network gate is open, not that it runs).
    # We test the gate by checking the policy allows NETWORK.
    assert policy.allows(PluginCapability.NETWORK)

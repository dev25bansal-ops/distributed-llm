"""Regression tests for M16 — P2P/IPFS plugin distribution backend.

M16 introduces a pluggable `PluginDistributionBackend` interface plus two
concrete backends (`LocalBackend`, `IpfsBackend` scaffold) into the plugin
marketplace, so sovereign / air-gapped deployments are not tied to PyPI.

These tests exercise the public interface:

* `LocalBackend.fetch` works against a real local artifact.
* `IpfsBackend` is selectable and its `fetch` resolves a gateway URL
  (urllib is monkeypatched to avoid any real network call).
* `IpfsBackend.publish` is an explicit SCAFFOLD stub (NotImplementedError).
* `PluginMarketplace` can be constructed with the P2P backend (by name or by
  env var) and exposes `fetch_plugin` / `distribution_backend`.

They must PASS after the fix and would FAIL on the pre-M16 code (no backend
classes, no `distribution` kwarg, no `fetch_plugin`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def local_artifact(tmp_path: Path) -> Path:
    art = tmp_path / "my-plugin.whl"
    art.write_bytes(b"fake-wheel-bytes")
    return art


# ── PluginDistributionBackend interface ──────────────────────────────


def test_local_backend_fetch_returns_real_path(local_artifact: Path):
    from distllm.core.plugin_marketplace import LocalBackend

    backend = LocalBackend()
    got = backend.fetch(str(local_artifact))
    assert Path(got) == local_artifact
    assert Path(got).read_bytes() == b"fake-wheel-bytes"


def test_local_backend_fetch_missing_raises():
    from distllm.core.plugin_marketplace import LocalBackend

    backend = LocalBackend()
    with pytest.raises(FileNotFoundError):
        backend.fetch("/nonexistent/path/plugin.whl")


def test_local_backend_publish_copies_into_base_dir(tmp_path: Path, local_artifact: Path):
    from distllm.core.plugin_marketplace import LocalBackend

    backend = LocalBackend(base_dir=str(tmp_path / "store"))
    backend.publish("my-plugin.whl", str(local_artifact))
    assert (tmp_path / "store" / "my-plugin.whl").read_bytes() == b"fake-wheel-bytes"


# ── IpfsBackend scaffold ─────────────────────────────────────────────


def test_ipfs_backend_is_selectable_and_scaffold_named():
    from distllm.core.plugin_marketplace import (
        IpfsBackend,
        PluginDistributionBackend,
        get_distribution_backend,
    )

    backend = get_distribution_backend("ipfs")
    assert isinstance(backend, IpfsBackend)
    assert isinstance(backend, PluginDistributionBackend)
    # Clearly marked SCAFFOLD in the docstring.
    assert "SCAFFOLD" in IpfsBackend.__doc__
    assert backend.name == "ipfs"


def test_ipfs_backend_fetch_resolves_gateway_url(monkeypatch, tmp_path: Path):
    import distllm.core.plugin_marketplace as pm

    captured: dict = {}

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(url, timeout=30):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResp(b"p2p-plugin-bytes")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    backend = pm.IpfsBackend(
        gateway="https://example-gateway.example/ipfs/",
        cache_dir=str(tmp_path / "ipfs_cache"),
    )
    out = backend.fetch("QmExampleCid123")
    assert captured["url"] == "https://example-gateway.example/ipfs/QmExampleCid123"
    assert Path(out).read_bytes() == b"p2p-plugin-bytes"


def test_ipfs_backend_default_gateway_from_env(monkeypatch, tmp_path: Path):
    import distllm.core.plugin_marketplace as pm

    monkeypatch.setenv("DISTLLM_IPFS_GATEWAY", "https://cloudflare-ipfs.com/ipfs/")
    captured: dict = {}

    class _FakeResp:
        def read(self):
            return b"x"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(url, timeout=30):
        captured["url"] = url
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    backend = pm.IpfsBackend(cache_dir=str(tmp_path / "c2"))
    assert backend.gateway == "https://cloudflare-ipfs.com/ipfs/"
    backend.fetch("QmAbc")
    assert captured["url"] == "https://cloudflare-ipfs.com/ipfs/QmAbc"


def test_ipfs_backend_publish_is_scaffold_stub():
    from distllm.core.plugin_marketplace import IpfsBackend

    backend = IpfsBackend(cache_dir=None)
    with pytest.raises(NotImplementedError) as exc:
        backend.publish("QmWhatever", "/tmp/artifact.whl")
    assert "SCAFFOLD" in str(exc.value)


# ── Marketplace wiring ───────────────────────────────────────────────


def test_marketplace_constructs_with_p2p_backend_by_name():
    from distllm.core.plugin_marketplace import (
        IpfsBackend,
        PluginMarketplace,
    )

    mp = PluginMarketplace(distribution="ipfs", cache_dir=None)
    assert isinstance(mp.distribution_backend, IpfsBackend)
    assert mp._distribution_name == "ipfs"


def test_marketplace_constructs_with_p2p_backend_via_env(monkeypatch):
    from distllm.core.plugin_marketplace import (
        IpfsBackend,
        PluginMarketplace,
    )

    monkeypatch.setenv("DISTLLM_PLUGIN_DIST", "ipfs")
    mp = PluginMarketplace(cache_dir=None)
    assert isinstance(mp.distribution_backend, IpfsBackend)


def test_marketplace_constructs_with_explicit_backend_instance(local_artifact: Path):
    from distllm.core.plugin_marketplace import (
        LocalBackend,
        PluginMarketplace,
    )

    backend = LocalBackend()
    mp = PluginMarketplace(distribution=backend)
    assert mp.distribution_backend is backend


def test_marketplace_fetch_plugin_routes_to_active_backend(local_artifact: Path):
    from distllm.core.plugin_marketplace import (
        LocalBackend,
        PluginMarketplace,
    )

    mp = PluginMarketplace(distribution=LocalBackend())
    got = mp.fetch_plugin(str(local_artifact))
    assert Path(got) == local_artifact


def test_marketplace_default_distribution_is_local():
    from distllm.core.plugin_marketplace import (
        LocalBackend,
        PluginMarketplace,
    )

    mp = PluginMarketplace()
    assert isinstance(mp.distribution_backend, LocalBackend)
    assert mp._distribution_name == "local"


def test_marketplace_unknown_backend_raises():
    from distllm.core.plugin_marketplace import PluginMarketplace

    with pytest.raises(ValueError):
        PluginMarketplace(distribution="bittorrent")

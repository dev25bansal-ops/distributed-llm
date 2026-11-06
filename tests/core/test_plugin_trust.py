"""SEC-3 regression tests: plugin installation is fail-closed and
signature-verifiable (core/plugin_trust.py).

- install_plugin / marketplace.install refuse without DISTLLM_PLUGIN_INSTALL_ALLOWED
- with the gate on, pip install proceeds (subprocess mocked)
- with DISTLLM_PLUGIN_SIGNING_PUBKEY set, a verified Ed25519 manifest is required
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.plugin_marketplace import PluginMarketplace
from distllm.core.plugin_system import PluginSystem
from distllm.core.plugin_trust import (
    INSTALL_ALLOWED_ENV,
    SIGNING_PUBKEY_ENV,
    install_allowed,
    verify_plugin_manifest,
)


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(INSTALL_ALLOWED_ENV, raising=False)
    monkeypatch.delenv(SIGNING_PUBKEY_ENV, raising=False)


def test_gate_disabled_by_default():
    assert install_allowed() is False


def test_install_plugin_refused_without_gate():
    system = PluginSystem()
    assert system.install_plugin("anything") is False


def test_marketplace_install_refused_without_gate():
    market = PluginMarketplace()
    assert market.install("anything") is False


def test_install_proceeds_with_gate(monkeypatch):
    monkeypatch.setenv(INSTALL_ALLOWED_ENV, "1")
    system = PluginSystem()
    with patch("subprocess.run") as run, patch.object(system, "discover_entry_points"):
        run.return_value = MagicMock(returncode=0, stderr="")
        assert system.install_plugin("coolplugin") is True
        cmd = run.call_args.args[0]
        assert cmd[-1] == "distllm-plugin-coolplugin"
        assert "-m" in cmd and "pip" in cmd


def test_unsafe_name_still_rejected_with_gate(monkeypatch):
    monkeypatch.setenv(INSTALL_ALLOWED_ENV, "1")
    system = PluginSystem()
    assert system.install_plugin("../evil") is False


def _write_signed_manifest(tmp_path, package_name, pubkey, signing_key, tamper=False):
    manifest = {"name": package_name.replace("distllm-plugin-", ""), "package": package_name}
    blob = json.dumps(manifest, indent=2).encode("utf-8")
    if tamper:
        blob = blob.replace(package_name.encode(), b"distllm-plugin-other")
    (tmp_path / f"{package_name}.manifest.json").write_bytes(blob)
    sig = signing_key.sign(blob)
    (tmp_path / f"{package_name}.manifest.json.sig").write_bytes(base64.b64encode(sig))


class TestSignedManifest:
    def _keys(self):
        pytest.importorskip("cryptography")
        from cryptography.hazmat.primitives.asymmetric import ed25519

        key = ed25519.Ed25519PrivateKey.generate()
        pub_pem = key.public_key().public_bytes(
            encoding=__import__("cryptography").hazmat.primitives.serialization.Encoding.PEM,
            format=__import__("cryptography").hazmat.primitives.serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return key, pub_pem

    def test_valid_manifest_verifies(self, tmp_path):
        key, pub_pem = self._keys()
        key_path = tmp_path / "pub.pem"
        key_path.write_bytes(pub_pem)
        _write_signed_manifest(tmp_path, "distllm-plugin-good", key_path, key)
        result = verify_plugin_manifest("good", "distllm-plugin-good", tmp_path, key_path)
        assert result.verified is True

    def test_tampered_manifest_fails(self, tmp_path):
        key, pub_pem = self._keys()
        key_path = tmp_path / "pub.pem"
        key_path.write_bytes(pub_pem)
        _write_signed_manifest(tmp_path, "distllm-plugin-good", key_path, key, tamper=True)
        result = verify_plugin_manifest("good", "distllm-plugin-good", tmp_path, key_path)
        assert result.verified is False

    def test_missing_manifest_fails_closed(self, tmp_path):
        key, pub_pem = self._keys()
        key_path = tmp_path / "pub.pem"
        key_path.write_bytes(pub_pem)
        result = verify_plugin_manifest("missing", "distllm-plugin-missing", tmp_path, key_path)
        assert result.verified is False

    def test_install_refused_when_key_set_but_manifest_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(INSTALL_ALLOWED_ENV, "1")
        monkeypatch.setenv(SIGNING_PUBKEY_ENV, str(tmp_path / "pub.pem"))
        system = PluginSystem()
        assert system.install_plugin("unpinned") is False

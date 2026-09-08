"""Regression tests for O-01: DISTLLM_DHT_REQUIRE_SECRET must be enforced.

The env var was documented at ``kademlia_dht.py:84`` with a full operator
policy block claiming "KademliaDHT.__init__ will refuse to start" when set
— but no code ever read it. A production operator setting the kill-switch
got zero behavior change: unauthenticated STORE mode proceeded with a
warning log.

Contract pinned here:
1. ``DISTLLM_DHT_REQUIRE_SECRET=1`` + no secret + ``allow_unauthenticated=True``
   → ``ConfigError`` at construction (the documented refusal).
2. Same env var but a secret IS configured → constructs normally.
3. Env var unset → legacy open mode still allowed with its warning (back-compat).
4. Env var set to something other than "1" → treated as unset (no accidental
   truthy parsing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import distllm.dist.p2p.kademlia_dht as kademlia_dht  # noqa: E402
from distllm.dist.p2p.kademlia_dht import KademliaDHT  # noqa: E402
from distllm.errors import ConfigError  # noqa: E402

ENV_REQUIRE = kademlia_dht.DHT_REQUIRE_SECRET_ENV_VAR
ENV_SECRET = kademlia_dht.DHT_SECRET_ENV_VAR


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_REQUIRE, raising=False)
    monkeypatch.delenv(ENV_SECRET, raising=False)


class TestRequireSecretEnforced:
    def test_env_var_refuses_unauthenticated_without_secret(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_REQUIRE, "1")
        with pytest.raises(ConfigError, match="DISTLLM_DHT_REQUIRE_SECRET"):
            KademliaDHT(allow_unauthenticated=True)

    def test_env_var_refuses_even_without_flag(self, monkeypatch) -> None:
        # The kill-switch must not be escapable by omitting the flag:
        # the default path already rejects STOREs, but construction should
        # still fail-loud so misconfiguration is visible at boot, not at
        # first rejected write.
        monkeypatch.setenv(ENV_REQUIRE, "1")
        with pytest.raises(ConfigError):
            KademliaDHT()

    def test_env_var_with_secret_constructs(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_REQUIRE, "1")
        monkeypatch.setenv(ENV_SECRET, "cluster-secret")
        dht = KademliaDHT(allow_unauthenticated=True)
        assert dht._shared_secret == "cluster-secret"

    def test_env_var_with_explicit_secret_arg_constructs(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_REQUIRE, "1")
        dht = KademliaDHT(shared_secret="explicit", allow_unauthenticated=True)
        assert dht._shared_secret == "explicit"

    def test_env_var_unset_allows_legacy_open_mode(self) -> None:
        dht = KademliaDHT(allow_unauthenticated=True)
        assert not dht._shared_secret
        assert dht._allow_unauthenticated is True

    def test_env_var_non_one_treated_as_unset(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_REQUIRE, "true")
        dht = KademliaDHT(allow_unauthenticated=True)  # must not raise
        assert dht._allow_unauthenticated is True


class TestStoreTokenStillWorks:
    """The enforcement must not break the existing HMAC store path."""

    def test_issue_and_verify_store_token_roundtrip(self) -> None:
        import time as _time

        dht = KademliaDHT(shared_secret="s3cret")
        expires = int(_time.time()) + 60
        token = dht._make_store_token("n1", "key-1", "deadbeef", expires)
        assert dht._verify_store_token("n1", "key-1", "deadbeef", token, expires) is True

    def test_wrong_key_rejects_token(self) -> None:
        import time as _time

        dht = KademliaDHT(shared_secret="s3cret")
        expires = int(_time.time()) + 60
        token = dht._make_store_token("n1", "key-1", "deadbeef", expires)
        assert dht._verify_store_token("n1", "key-2", "deadbeef", token, expires) is False

    def test_no_secret_rejects_unauthenticated_store(self) -> None:
        dht = KademliaDHT()  # fail-closed default, no env vars
        assert dht._verify_store_token("n1", "k", "v", "", None) is False

    def test_expired_token_rejected(self) -> None:
        dht = KademliaDHT(shared_secret="s3cret")
        stale = 1_000_000_000  # long past
        token = dht._make_store_token("n1", "k", "v", stale)
        assert dht._verify_store_token("n1", "k", "v", token, stale) is False

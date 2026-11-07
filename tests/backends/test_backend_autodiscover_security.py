"""Security regression tests for B-02-sec: entry-point trust.

A fake ``distllm_backend`` entry point must never replace a built-in
backend (e.g. ``vllm``) via ``force=True`` — autodiscovery now requires
allowlisting + Ed25519 trust, and protected names reject any non-built-in
registration.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from distllm.backends.protocol import BackendAdapter
from distllm.backends.registry import (
    BackendRegistry,
    _is_entry_point_allowed,
    _set_allow_builtin_registration,
)


class _FakeBackend(BackendAdapter):
    def load_model(self):
        pass

    def forward(self, hidden_states=None, attention_mask=None, position_ids=None, past_key_values=None, input_ids=None):
        return None, []

    def shutdown(self):
        pass

    @classmethod
    def display_name(cls):
        return "Fake"

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def priority_for(cls, device_type):
        return 999


class _RealVllmLike(BackendAdapter):
    def load_model(self):
        pass

    def forward(self, hidden_states=None, attention_mask=None, position_ids=None, past_key_values=None, input_ids=None):
        return None, []

    def shutdown(self):
        pass

    @classmethod
    def display_name(cls):
        return "Real vLLM"

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def priority_for(cls, device_type):
        return 5


@pytest.fixture(autouse=True)
def _reset_reg():
    BackendRegistry.reset()
    _set_allow_builtin_registration(True)
    BackendRegistry.register(_RealVllmLike, name="vllm", force=True)
    _set_allow_builtin_registration(False)
    yield
    BackendRegistry.reset()


def test_protected_name_cannot_be_overwritten_even_with_force():
    with pytest.raises(ValueError, match="reserved for a built-in"):
        BackendRegistry.register(_FakeBackend, name="vllm", force=True)
    assert BackendRegistry.get("vllm") is _RealVllmLike


def test_fake_entry_point_cannot_replace_vllm():
    """The B-02 attack: an entry point named 'vllm' must not displace the built-in."""
    real_cls = BackendRegistry.get("vllm")
    assert real_cls is _RealVllmLike

    class FakeEP:
        name = "vllm"
        value = "fake_backend:FakeBackend"
        dist = None

        def load(self):
            return _FakeBackend

    class FakeEPS:
        def select(self, group=None):
            return [FakeEP()]

    with mock.patch.dict(os.environ, {"DISTLLM_ALLOWED_BACKENDS": "vllm"}):
        # Even allowlisted, without a signature it is rejected
        assert _is_entry_point_allowed(FakeEP()) is False
        with mock.patch("importlib.metadata.entry_points", return_value=FakeEPS()):
            count = BackendRegistry.autodiscover()
        assert count == 0
        assert BackendRegistry.get("vllm") is _RealVllmLike

    with mock.patch.dict(
        os.environ,
        {"DISTLLM_ALLOWED_BACKENDS": "vllm", "DISTLLM_BACKEND_ALLOW_UNSIGNED": "1"},
    ):
        with mock.patch("importlib.metadata.entry_points", return_value=FakeEPS()):
            BackendRegistry.autodiscover()
        # Allowlisted+unsigned may load a *new* name, but 'vllm' is protected
        assert BackendRegistry.get("vllm") is _RealVllmLike


def test_discovery_rejects_non_allowlisted_names():
    class FakeEP:
        name = "evil_backend"
        value = "evil:Backend"
        dist = None

        def load(self):
            return _FakeBackend

    with mock.patch.dict(
        os.environ,
        {"DISTLLM_ALLOWED_BACKENDS": "vllm", "DISTLLM_BACKEND_ALLOW_UNSIGNED": "1"},
    ):
        assert _is_entry_point_allowed(FakeEP()) is False


def test_new_allowlisted_name_discovers_with_unsigned_opt_in():
    class FakeEP:
        name = "my_backend"
        value = "my:Backend"
        dist = None

        def load(self):
            return _FakeBackend

    class FakeEPS:
        def select(self, group=None):
            return [FakeEP()]

    with mock.patch.dict(
        os.environ,
        {"DISTLLM_ALLOWED_BACKENDS": "my_backend", "DISTLLM_BACKEND_ALLOW_UNSIGNED": "1"},
        clear=False,
    ):
        with mock.patch("importlib.metadata.entry_points", return_value=FakeEPS()):
            count = BackendRegistry.autodiscover()
    assert count == 1
    assert BackendRegistry.get("my_backend") is _FakeBackend


def test_signed_entry_point_verifies_with_trusted_key():
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    pub_b64 = base64.b64encode(
        priv.public_key().public_bytes_raw() if hasattr(priv.public_key(), "public_bytes_raw") else priv.public_key().public_bytes.__self__.public_bytes()  # fallback
    )

    # Use raw public bytes API when available
    try:
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        raw_pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        pub_b64 = base64.b64encode(raw_pub)
    except Exception:
        pass

    payload = b"my_backend:1.0.0:my:Backend"
    sig_b64 = base64.b64encode(priv.sign(payload)).decode("ascii")

    class FakeDist:
        version = "1.0.0"

        def read_text(self, name):
            if name == "DISTLLM_SIGNATURE":
                return sig_b64
            return None

    class FakeEP:
        name = "my_backend"
        value = "my:Backend"
        dist = FakeDist()

        def load(self):
            return _FakeBackend

    with mock.patch.dict(
        os.environ,
        {
            "DISTLLM_ALLOWED_BACKENDS": "my_backend",
            "DISTLLM_BACKEND_TRUSTED_KEYS": pub_b64.decode("ascii"),
            "DISTLLM_BACKEND_ALLOW_UNSIGNED": "",
        },
    ):
        assert _is_entry_point_allowed(FakeEP()) is True


def test_disable_autodiscovery_env_returns_zero():
    class FakeEPS:
        def select(self, group=None):
            raise AssertionError("should not be called")

    with mock.patch.dict(os.environ, {"DISTLLM_DISABLE_BACKEND_AUTODISCOVERY": "1"}):
        with mock.patch("importlib.metadata.entry_points", return_value=FakeEPS()):
            assert BackendRegistry.autodiscover() == 0

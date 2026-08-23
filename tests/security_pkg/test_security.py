"""Tests for distllm.security: URL safety, watermark config, and model watermarking.

Covers:
  - ``WatermarkConfig``   defaults, custom values, string-to-bytes, greenlist derivation
  - ``ModelWatermark``    construction and facade delegation
  - ``validate_http_url`` public IP, private IP, invalid scheme, missing hostname
  - ``safe_urlopen``      URL-level validation (before DNS/network)
  - ``hf_revision``       env-var resolution, fallback, and strict-mode error

No MagicMock — uses monkeypatch and inline attribute-replacement stubs for socket.
"""

from __future__ import annotations

import socket

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_utils_mod = load_module("distllm/security/utils.py")
_watermark_mod = load_module("distllm/security/watermark.py")

safe_urlopen = _utils_mod.safe_urlopen
validate_http_url = _utils_mod.validate_http_url
hf_revision = _utils_mod.hf_revision
WatermarkConfig = _watermark_mod.WatermarkConfig
ModelWatermark = _watermark_mod.ModelWatermark
WatermarkError = _watermark_mod.WatermarkError


# ===================================================================
# WatermarkConfig TESTS
# ===================================================================


class TestWatermarkConfig:
    """WatermarkConfig dataclass -- defaults, custom values, field behavior."""

    def test_defaults(self) -> None:
        """A plain WatermarkConfig() should generate a secret_key and derive a greenlist_key."""
        cfg = WatermarkConfig()
        assert isinstance(cfg.secret_key, bytes)
        assert len(cfg.secret_key) == 32
        assert isinstance(cfg.greenlist_key, bytes)
        assert len(cfg.greenlist_key) == 32
        assert cfg.target_fraction == 0.0001
        assert cfg.greenlist_fraction == 0.25
        assert cfg.gumbel_temperature == 1.0
        assert cfg.message == ""
        assert cfg.signing_key is None
        assert cfg.verify_key is None

    def test_custom_values(self) -> None:
        """Custom keyword arguments should override WatermarkConfig defaults."""
        cfg = WatermarkConfig(
            secret_key=b"my-fixed-32-byte-secret-key!!!!",
            message="(c) 2026 Acme Corp",
            target_fraction=0.001,
            greenlist_fraction=0.5,
            gumbel_temperature=2.0,
        )
        assert cfg.secret_key == b"my-fixed-32-byte-secret-key!!!!"
        assert cfg.message == "(c) 2026 Acme Corp"
        assert cfg.target_fraction == 0.001
        assert cfg.greenlist_fraction == 0.5
        assert cfg.gumbel_temperature == 2.0

    def test_secret_key_string_conversion(self) -> None:
        """A string secret_key should be encoded to bytes in __post_init__."""
        cfg = WatermarkConfig(secret_key="my-string-key")
        assert isinstance(cfg.secret_key, bytes)
        assert cfg.secret_key == b"my-string-key"

    def test_greenlist_key_derivation(self) -> None:
        """When greenlist_key is omitted it should be deterministically derived from secret_key."""
        cfg = WatermarkConfig(secret_key=b"test-secret-32-bytes-for-derivation")
        assert cfg.greenlist_key is not None
        assert len(cfg.greenlist_key) == 32

        cfg2 = WatermarkConfig(secret_key=b"test-secret-32-bytes-for-derivation")
        assert cfg.greenlist_key == cfg2.greenlist_key


# ===================================================================
# ModelWatermark TESTS
# ===================================================================


class TestModelWatermark:
    """ModelWatermark construction and facade delegation."""

    def test_construction_default(self) -> None:
        """A default ModelWatermark should create sub-watermarks with default config."""
        mw = ModelWatermark()
        assert mw._config is not None
        assert mw._weight_wm is not None
        assert mw._gumbel_wm is not None
        assert isinstance(mw._config, WatermarkConfig)

    def test_construction_with_config(self) -> None:
        """A custom WatermarkConfig should be shared by both sub-watermarks."""
        cfg = WatermarkConfig(secret_key=b"custom-config-key-32-bytes!!", message="test")
        mw = ModelWatermark(config=cfg)
        assert mw._config is cfg
        assert mw._weight_wm._config is cfg
        assert mw._gumbel_wm._config is cfg


# ===================================================================
# validate_http_url TESTS
# ===================================================================


class TestValidateHttpUrl:
    """URL validation for SSRF / DNS-rebinding protection."""

    def test_valid_public_url(self) -> None:
        """A valid public HTTPS URL should pass validation and return the URL unchanged."""
        original_getaddrinfo = _utils_mod.socket.getaddrinfo

        def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

        _utils_mod.socket.getaddrinfo = _fake_getaddrinfo
        try:
            result = validate_http_url("https://example.com")
            assert result == "https://example.com"
        finally:
            _utils_mod.socket.getaddrinfo = original_getaddrinfo

    def test_invalid_scheme(self) -> None:
        """A non-HTTP(S) scheme should raise ValueError."""
        with pytest.raises(ValueError, match="scheme"):
            validate_http_url("ftp://example.com")

    def test_no_hostname(self) -> None:
        """A URL without a hostname should raise ValueError."""
        with pytest.raises(ValueError, match="hostname"):
            validate_http_url("http://")

    def test_private_ip_rejected(self) -> None:
        """A URL resolving to a private IP should raise ValueError."""
        original_getaddrinfo = _utils_mod.socket.getaddrinfo

        def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80))]

        _utils_mod.socket.getaddrinfo = _fake_getaddrinfo
        try:
            with pytest.raises(ValueError, match="non-public"):
                validate_http_url("http://internal.example.com")
        finally:
            _utils_mod.socket.getaddrinfo = original_getaddrinfo


# ===================================================================
# safe_urlopen TESTS
# ===================================================================


class TestSafeUrlOpen:
    """safe_urlopen URL-level validation (before any DNS/network call)."""

    def test_invalid_scheme(self) -> None:
        """An invalid scheme on safe_urlopen should raise ValueError without network I/O."""
        with pytest.raises(ValueError, match="scheme"):
            safe_urlopen("ftp://example.com", timeout=5.0)

    def test_no_hostname(self) -> None:
        """A URL without a hostname should raise ValueError."""
        with pytest.raises(ValueError, match="hostname"):
            safe_urlopen("http:///path", timeout=5.0)


# ===================================================================
# hf_revision TESTS
# ===================================================================


class TestHfRevision:
    """Model revision pinning via ``hf_revision()``."""

    def test_default_returns_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without any env var set, hf_revision() should return 'main' with a warning."""
        monkeypatch.delenv("DISTLLM_MODEL_REVISION", raising=False)
        monkeypatch.delenv("HF_MODEL_REVISION", raising=False)
        monkeypatch.delenv("DISTLLM_REQUIRE_MODEL_REVISION", raising=False)
        with pytest.warns(UserWarning, match="unpinned"):
            result = hf_revision()
        assert result == "main"

    def test_env_var_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DISTLLM_MODEL_REVISION should be returned when set."""
        monkeypatch.setenv("DISTLLM_MODEL_REVISION", "abc123def")
        result = hf_revision()
        assert result == "abc123def"

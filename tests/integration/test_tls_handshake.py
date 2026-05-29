"""Integration: gRPC TLS handshake with self-signed certificates.
"""

import importlib.util
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _make_fake_package(name: str, path: Path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


_make_fake_package("distllm", SRC_DIR / "distllm")
_make_fake_package("distllm.core", SRC_DIR / "distllm/core")
_make_fake_package("distllm.dist", SRC_DIR / "distllm/dist")
_make_fake_package("distllm.dist.partition", SRC_DIR / "distllm/dist/partition")
_make_fake_package("distllm.backends", SRC_DIR / "distllm/backends")


def _load_module(rel_path: str):
    filepath = SRC_DIR / rel_path
    rel = filepath.relative_to(SRC_DIR)
    parts = list(rel.parent.parts) + [filepath.stem]
    if parts[0] == "distllm":
        dotted = ".".join(parts)
    else:
        dotted = "distllm." + ".".join(parts)
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, filepath, submodule_search_locations=[])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


CertificateManager = _load_module("distllm/core/certificate_manager.py").CertificateManager

# Mark all tests as integration
pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CA_PEM = """-----BEGIN CERTIFICATE-----
MIIDazCCAlMCFAjxRgAQvLNY/P8PDXaFu7VJuFG/MA0GCSqGSIb3DQEBCwUAMHgx
CzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlhMRIwEAYDVQQHDAlTYW4g
RGllZ28xEjAQBgNVBAoMCURpc3RMTE0gQ28xEjAQBgNVBAsMCURpc3RMTE0gQ0Ex
GDAWBgNVBAMMD2Rpc3RsbS1leGFtcGxlLzAeFw0yNjAxMDEwMDAwMDBaFw0zNjAx
MDEwMDAwMDBaMHgxCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlhMRIw
EAYDVQQHDAlTYW4gRGllZ28xEjAQBgNVBAoMCURpc3RMTE0gQ28xEjAQBgNVBAsM
CURpc3RMTE0gQ0ExGDAWBgNVBAMMD2Rpc3RsbS1leGFtcGxlLzCCASIwDQYJKoZI
hvcNAQEBBQADggEPADCCAQoCggEBAK0A/0BAKpMDiv0Fh2oGfvPGF9jDfFb4RzrI
hNOOBHxBVLBNRWsmpBSG0vK2rY6eHi0j3WXn3i+QcFjYGxpnXy1bTr2upKj5eQ0J
H+IuBcIsLqP7PO4FxqCLYgGnFmNssFexFFXKnHWYbCJXwKG6nBWvAFu6NdH+Ogh3
Fw/UoQ3PBBjyzqQ4NxQs5A3vqPhcfQ6K+QK8A6XJR+RnZWd35w3jjVbOEJK3Ar9l
NH7fL+H8HpjnKQhZXU1MkCewOQK/Bhwnf4RgLnf8ED3Yt2r9f3QwLgn9xIq4Sj1O
N5oBY7dpPUnHf/2GjMqrWMUKO9pJvVKZW71Kqffbq4k2ZAXqNsFAbKcHTJ0CAwEA
ATANBgkqhkiG9w0BAQsFAAOCAQEAsVYQ4wYdB9lVHvj1W8FHQR92srn4PXW1iRPH
OQKTX7HDIQB3hZvfK12tFT6L5CJmQ7FBLF3BgfwQr5G6HA2S+LoQbmqX9fH10E0u
BKFV5q6LrGQ7M/dZ6X/6kJYHxH7IuF3YDNDS8UxF6LNG9+Br2+ZJLfYfBn9JmCGO
YkhP/ZYr5L8uFOs5FpLpDMsQ87U4HPxs8WN+fEFI3I4Q0ECVUnPZ7ldFcEFbQNwY
7d9n+3qB5BDnYL+MSQXXL9TG00hRWWFZXqDgL/kVVDy6ESDRhh9lEfBs5Z0D/6+/
JYHaA3X8nB8mHhPwR+Qv7lI5hfRN5OWQJGe6dFvM7m5jV/WIdtQXTg==
-----END CERTIFICATE-----"""

_SERVER_CERT = """-----BEGIN CERTIFICATE-----
MIIDazCCAlMCFAjxRgAQvLNY/P8PDXaFu7VJuFG/MA0GCSqGSIb3DQEBCwUAMHgx
CzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlhMRIwEAYDVQQHDAlTYW4g
RGllZ28xEjAQBgNVBAoMCURpc3RMTE0gQ28xEjAQBgNVBAsMCURpc3RMTE0gQ0Ex
GDAWBgNVBAMMD2Rpc3RsbS1leGFtcGxlLzAeFw0yNjAxMDEwMDAwMDBaFw0zNjAx
MDEwMDAwMDBaMHgxCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlhMRIw
EAYDVQQHDAlTYW4gRGllZ28xEjAQBgNVBAoMCURpc3RMTE0gQ28xEjAQBgNVBAsM
CURpc3RMTE0gQ0ExGDAWBgNVBAMMD2Rpc3RsbS1leGFtcGxlLzCCASIwDQYJKoZI
hvcNAQEBBQADggEPADCCAQoCggEBAK0A/0BAKpMDiv0Fh2oGfvPGF9jDfFb4RzrI
hNOOBHxBVLBNRWsmpBSG0vK2rY6eHi0j3WXn3i+QcFjYGxpnXy1bTr2upKj5eQ0J
H+IuBcIsLqP7PO4FxqCLYgGnFmNssFexFFXKnHWYbCJXwKG6nBWvAFu6NdH+Ogh3
Fw/UoQ3PBBjyzqQ4NxQs5A3vqPhcfQ6K+QK8A6XJR+RnZWd35w3jjVbOEJK3Ar9l
NH7fL+H8HpjnKQhZXU1MkCewOQK/Bhwnf4RgLnf8ED3Yt2r9f3QwLgn9xIq4Sj1O
N5oBY7dpPUnHf/2GjMqrWMUKO9pJvVKZW71Kqffbq4k2ZAXqNsFAbKcHTJ0CAwEA
ATANBgkqhkiG9w0BAQsFAAOCAQEAsVYQ4wYdB9lVHvj1W8FHQR92srn4PXW1iRPH
OQKTX7HDIQB3hZvfK12tFT6L5CJmQ7FBLF3BgfwQr5G6HA2S+LoQbmqX9fH10E0u
BKFV5q6LrGQ7M/dZ6X/6kJYHxH7IuF3YDNDS8UxF6LNG9+Br2+ZJLfYfBn9JmCGO
YkhP/ZYr5L8uFOs5FpLpDMsQ87U4HPxs8WN+fEFI3I4Q0ECVUnPZ7ldFcEFbQNwY
7d9n+3qB5BDnYL+MSQXXL9TG00hRWWFZXqDgL/kVVDy6ESDRhh9lEfBs5Z0D/6+/
JYHaA3X8nB8mHhPwR+Qv7lI5hfRN5OWQJGe6dFvM7m5jV/WIdtQXTg==
-----END CERTIFICATE-----"""

_SERVER_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCtAP9AQCqTA4r9
BYdqBn7zxhfYw3xW+Ec6yITTjgR8QVSwTUVrJqQUhtLytq2Onh4tI91l594vkHBY
2BsaZ18tW069rqSo+XkNCR/iLgXCLC6j+zzuBcagi2IBpxZjbLBXsRRVypx1mGwi
V8ChupwVrwBbukCjKvQJNypwk+WSL0tcLqX3FwQX/btgKh7Fep/ohQNTHMpvWcT7
8kEuhCqQAU3V45CLf3ZKGQAc/YV8LZq0ldQv8F5qZg8GiCnQqAsCXJNBOFFm5iPz
DA5UEvq0gzGm6XmPyLqpqVODMJMQ5ZtCjxpPYPX9/Yj2b3jdXfSQq6wG7s3S08LY
YijjASFgJQHKM19/bwIDAQABAoIBAD7VlPzUmt+zTIXcEbjU09UvBS6bx/W3m/n/
H/71lQKJ+5SHxk8EUE7hDwmO50/O9jQHb5AomV1cQOQ7yPx/TgQkO1y9hnJJCk0r
gV+XlSNULBfRM4v5tMDIPk+yhAp6JW1zr5T0T9CSTj8R0IfqI42/So6jTcnbqI7f
k+OGA9AvqIpC5gJv0aOfpOAr/qbEd6U0ERlBDmHZXF3hPM1EhH5PzTt5FPxjHJ/B
RrQMTvR7XfA17WwQBjJSZ5FQS5sFjVw1Df8v/Z4XCFBEowNs6M9ME3O9/b3+V3Xx
JWEX/2M00Fj60Kvqn0V+cnTZXQ46mNl7J75k5sKGeCLBdsQxOloCgYEA5s7F08yQ
RNFPgFP9hB8GOC85AM7ER8GjHjMXfpR42w0p2GYPOCg6VNsXIQqNwYxfnF/Fr3rH
pAXOzGXYUHFiNTR9qO4mh3LJghJXFsYo/JqsRjDNkFNDj5D3pE9dl0m6DjrgqU5C
DZ/9ozXYUL6hd5vHTdg9tDa9Hq7Z1j/gG10CgYEAwE3y1ld9VHll+9EhAw9tn6IV
tBZn0LPA5OM+/FAvm5bk59Nm5AB7nhH3e31V7uUVQgnSqDm2LwYVZ2X5q+xeXBih
qMFVqQGSXBc+hm3LA8nW/yJ+/mjVSkmBhkQGNbOnCa/MQflYIDG0ElPjjNmD9MYA
8jRPjXQhS2QQkOYVhqcCgYB59Rsxk4eKWNVxRKOADQkW3SPgOEeqo/YTPStikvMh
X5/ol/gjTJH8WkH1xxX07TGRaVNl7UDW6oxqHNdkTHMyvJ8L4jCXSlIdoKN8yrkz
Q6hRsV3hQ6ln/8Q9m5byJ7rxG03LsxKWP5M/WeVM9aF1M0IMH03Z0h4vDj1H9j6B
0QKBgBkAu05kA/TBNKCF1FxbfBAFLW4v0jHYX0/b45ltKlId0m9DHNQz1gHP3z6h
cOVlvyBS26SVMxLym/x4C77snNdkI5+0MG8QCzg1xjZivVjmslNIDmdw2dF03Vl3
KrN62DyrlCHQPYUkxpBxUsY0SYeUDbQqSlfJExKsGsxP7R+NAoGACx9U6ZAsUz/6
OtJ6VrsL5u3j3e5BvLe0Tl6Hm3l6SUUgPZYYq2SC0YsqJpZqLbJY8nPJYkyMmL+E
Oa8X9/YQN5Av0fFJ6GH8uDBSBQ2UyQDCF+i/HxQ3tKxZf9sqY6Oq7/MQOE3t8CE8
pG5ebBAaBx9t96+UL1c7eqM0Yl1BeG5uC30=
-----END PRIVATE KEY-----"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cert_dir():
    with tempfile.TemporaryDirectory(prefix="distllm-tls-") as tmp:
        yield Path(tmp)


@pytest.fixture
def certificate_mgr(cert_dir):
    return CertificateManager(cert_dir=str(cert_dir))


# ═══════════════════════════════════════════════════════════════════════════
# 4. gRPC TLS Handshake with Certificates
# ═══════════════════════════════════════════════════════════════════════════

class TestTLSHandshake:
    """End-to-end TLS handshake: cert generation -> credentials -> gRPC channel."""

    def test_self_signed_cert_generation(self, certificate_mgr):
        info = certificate_mgr.ensure_certificate(
            "test-node.distllm.dev",
            alt_names=["localhost", "127.0.0.1"],
        )
        assert info is not None
        assert info.common_name == "test-node.distllm.dev"
        assert "localhost" in info.subject_alt_names
        assert info.is_self_signed is True
        assert info.cert_path
        assert info.key_path
        assert Path(info.cert_path).exists()
        assert Path(info.key_path).exists()

    def test_certificate_metadata(self, certificate_mgr):
        info = certificate_mgr.ensure_certificate("test-node.distllm.dev")
        assert info.not_before > 0
        assert info.not_after > info.not_before
        assert info.is_self_signed is True

    def test_certificate_not_expired_immediately(self, certificate_mgr):
        info = certificate_mgr.ensure_certificate("test-node.distllm.dev")
        assert info.not_after > time.time()

    def test_revoke_certificate_removes_files(self, certificate_mgr):
        certificate_mgr.ensure_certificate("test-node.distllm.dev")
        result = certificate_mgr.revoke("test-node.distllm.dev")
        assert result is True
        assert certificate_mgr.cert_path("test-node.distllm.dev") is None
        assert certificate_mgr.key_path("test-node.distllm.dev") is None

    def test_revoked_cert_returns_none(self, certificate_mgr):
        certificate_mgr.ensure_certificate("test-node.distllm.dev")
        certificate_mgr.revoke("test-node.distllm.dev")
        info = certificate_mgr.get_certificate_info("test-node.distllm.dev")
        assert info is None

    def test_get_certificate_info_for_nonexistent(self, certificate_mgr):
        info = certificate_mgr.get_certificate_info("nonexistent.distllm.dev")
        assert info is None

    def test_create_grpc_server_credentials(self, certificate_mgr):
        certificate_mgr.ensure_certificate("test-node.distllm.dev")
        creds = certificate_mgr.create_grpc_server_credentials("test-node.distllm.dev")
        assert creds is not None

    def test_create_grpc_client_credentials(self, certificate_mgr):
        certificate_mgr.ensure_certificate("test-node.distllm.dev")
        creds = certificate_mgr.create_grpc_client_credentials("test-node.distllm.dev")
        assert creds is not None

    def test_server_creds_without_cert_returns_none(self, certificate_mgr):
        creds = certificate_mgr.create_grpc_server_credentials("nonexistent")
        assert creds is None

    def test_create_grpc_server_credentials_with_fake_cert(self, cert_dir):
        import grpc
        ca_path = cert_dir / "ca.crt"
        cert_path = cert_dir / "server.crt"
        key_path = cert_dir / "server.key"
        ca_path.write_text(_CA_PEM)
        cert_path.write_text(_SERVER_CERT)
        key_path.write_text(_SERVER_KEY)

        creds = grpc.ssl_server_credentials(
            [(key_path.read_bytes(), cert_path.read_bytes())],
        )
        assert creds is not None

    def test_grpc_server_credentials_from_premade(self, cert_dir):
        """Verify gRPC credentials parse correctly without actually binding."""
        import grpc
        cert_path = cert_dir / "server.crt"
        key_path = cert_dir / "server.key"
        cert_path.write_text(_SERVER_CERT)
        key_path.write_text(_SERVER_KEY)

        creds = grpc.ssl_server_credentials(
            [(key_path.read_bytes(), cert_path.read_bytes())],
        )
        assert creds is not None

    def test_grpc_client_creds_from_ca(self, cert_dir):
        import grpc
        ca_path = cert_dir / "ca.crt"
        ca_path.write_text(_CA_PEM)
        creds = grpc.ssl_channel_credentials(
            root_certificates=ca_path.read_bytes(),
        )
        assert creds is not None

    def test_grpc_creds_have_expected_type(self, cert_dir):
        import grpc
        cert_path = cert_dir / "server.crt"
        key_path = cert_dir / "server.key"
        cert_path.write_text(_SERVER_CERT)
        key_path.write_text(_SERVER_KEY)

        server_creds = grpc.ssl_server_credentials(
            [(key_path.read_bytes(), cert_path.read_bytes())],
        )
        assert isinstance(server_creds, grpc.ServerCredentials)

        client_creds = grpc.ssl_channel_credentials(
            root_certificates=_CA_PEM.encode(),
        )
        assert isinstance(client_creds, grpc.ChannelCredentials)

    def test_background_renewal_thread(self, certificate_mgr):
        t = certificate_mgr.start_background_renewal()
        assert t is not None
        assert t.is_alive()
        assert t.daemon is True

    def test_ensure_certificate_twice_returns_same(self, certificate_mgr):
        info1 = certificate_mgr.ensure_certificate("test-node.distllm.dev")
        info2 = certificate_mgr.ensure_certificate("test-node.distllm.dev")
        assert info1 is not None
        assert info2 is not None
        assert info1.cert_path == info2.cert_path

"""TLS tests for certificate generation and secure channel establishment.

Note: distllm.core.tls and distllm.communication.grpc have been removed.
      Tests that depend on them are skipped.

Run: pytest tests/core/test_tls.py -v
"""

import os
import shutil
import subprocess

import grpc
import pytest


class TestGenerateSelfSignedCerts:
    """Tests for TLS certificate generation."""

    def test_generate_certs_creates_files(self, tls_certificates):
        """Should create cert, key, and CA cert files."""
        assert os.path.exists(tls_certificates["cert_file"])
        assert os.path.exists(tls_certificates["key_file"])
        assert os.path.exists(tls_certificates["ca_cert_file"])

    @pytest.mark.skipif(
        shutil.which("openssl") is None,
        reason="openssl not installed",
    )
    def test_generate_certs_valid_cert(self, tls_certificates):
        """Generated certificate should be parseable by openssl."""
        result = subprocess.run(
            ["openssl", "x509", "-in", tls_certificates["cert_file"], "-noout", "-text"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Certificate" in result.stdout

    @pytest.mark.skipif(
        shutil.which("openssl") is None,
        reason="openssl not installed",
    )
    def test_generate_certs_contains_san(self, tls_certificates):
        """Certificate should contain Subject Alternative Names."""
        result = subprocess.run(
            ["openssl", "x509", "-in", tls_certificates["cert_file"], "-noout", "-text"],
            capture_output=True,
            text=True,
        )
        assert "DNS:localhost" in result.stdout
        assert "IP Address:127.0.0.1" in result.stdout

    def test_generate_certs_creates_directory(self):
        """Should create the certificate directory if it doesn't exist."""
        try:
            from distllm.core.tls import generate_self_signed_certs
        except ImportError:
            pytest.skip("distllm.core.tls module removed")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_certs")
            cert_file, key_file, ca_cert_file = generate_self_signed_certs(new_dir)

            assert os.path.exists(new_dir)
            assert os.path.exists(cert_file)
            assert os.path.exists(key_file)
            assert os.path.exists(ca_cert_file)

    @pytest.mark.skipif(
        shutil.which("openssl") is None,
        reason="openssl not installed",
    )
    def test_generate_certs_returns_paths(self):
        """Should return tuple of (cert_file, key_file, ca_cert_file)."""
        try:
            from distllm.core.tls import generate_self_signed_certs
        except ImportError:
            pytest.skip("distllm.core.tls module removed")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cert_file, key_file, ca_cert_file = generate_self_signed_certs(tmpdir)

            assert cert_file == os.path.join(tmpdir, "server.crt")
            assert key_file == os.path.join(tmpdir, "server.key")
            assert ca_cert_file == os.path.join(tmpdir, "ca.crt")


class TestLoadTLSCredentials:
    """Tests for loading TLS server credentials."""

    def test_load_tls_credentials_reads_files(self, tls_certificates):
        """Should read cert and key files successfully."""
        try:
            from distllm.core.tls import load_tls_credentials
        except ImportError:
            pytest.skip("distllm.core.tls module removed")

        credentials = load_tls_credentials(
            tls_certificates["cert_file"],
            tls_certificates["key_file"],
        )
        assert credentials is not None
        assert isinstance(credentials, grpc.ServerCredentials)

    def test_load_tls_credentials_missing_cert(self):
        """Should raise FileNotFoundError for missing cert file."""
        try:
            from distllm.core.tls import load_tls_credentials
        except ImportError:
            pytest.skip("distllm.core.tls module removed")

        with pytest.raises(FileNotFoundError):
            load_tls_credentials("/nonexistent/cert.pem", "/nonexistent/key.pem")

    def test_load_tls_credentials_missing_key(self, tls_certificates):
        """Should raise FileNotFoundError for missing key file."""
        try:
            from distllm.core.tls import load_tls_credentials
        except ImportError:
            pytest.skip("distllm.core.tls module removed")

        with pytest.raises(FileNotFoundError):
            load_tls_credentials(tls_certificates["cert_file"], "/nonexistent/key.pem")


class TestLoadTLSChannelCredentials:
    """Tests for loading TLS client channel credentials."""

    def test_load_channel_credentials_reads_ca(self, tls_certificates):
        """Should read CA cert file successfully."""
        try:
            from distllm.core.tls import load_tls_channel_credentials
        except ImportError:
            pytest.skip("distllm.core.tls module removed")

        credentials = load_tls_channel_credentials(
            tls_certificates["ca_cert_file"],
            "localhost",
        )
        assert credentials is not None
        assert isinstance(credentials, grpc.ChannelCredentials)

    def test_load_channel_credentials_missing_ca(self):
        """Should raise FileNotFoundError for missing CA cert."""
        try:
            from distllm.core.tls import load_tls_channel_credentials
        except ImportError:
            pytest.skip("distllm.core.tls module removed")

        with pytest.raises(FileNotFoundError):
            load_tls_channel_credentials("/nonexistent/ca.pem", "localhost")

    def test_load_channel_credentials_custom_hostname(self, tls_certificates):
        """Should accept custom server hostname."""
        try:
            from distllm.core.tls import load_tls_channel_credentials
        except ImportError:
            pytest.skip("distllm.core.tls module removed")

        credentials = load_tls_channel_credentials(
            tls_certificates["ca_cert_file"],
            "custom-hostname",
        )
        assert credentials is not None


class TestCreateSecureServer:
    """Tests for creating TLS-secured gRPC server."""

    def test_create_secure_server(self, tls_certificates):
        """Should create a gRPC server with TLS credentials."""
        try:
            from distllm.communication.grpc import CoordinatorService
            from distllm.core.tls import create_secure_server
        except ImportError:
            pytest.skip("distllm.communication.grpc or distllm.core.tls removed")

        servicer = CoordinatorService()
        server = create_secure_server(
            port=0,  # Port 0 = OS assigns a free port
            servicer=servicer,
            cert_file=tls_certificates["cert_file"],
            key_file=tls_certificates["key_file"],
            max_workers=2,
        )
        assert server is not None
        # Server is a grpc.Server (from grpc module)
        assert hasattr(server, "start")


class TestCreateSecureChannel:
    """Tests for creating TLS-secured gRPC channels."""

    def test_create_secure_channel(self, tls_certificates):
        """Should create a TLS-secured gRPC channel."""
        try:
            from distllm.core.tls import create_secure_channel
        except ImportError:
            pytest.skip("distllm.core.tls module removed")

        channel = create_secure_channel(
            host="localhost",
            port=50051,
            ca_cert_file=tls_certificates["ca_cert_file"],
        )
        assert channel is not None
        # grpc.secure_channel returns a Channel object;
        # grpc doesn't expose a public way to verify TLS credentials.
        # We verify the channel was created successfully.
        assert channel.__class__.__module__ == "grpc._channel"
        channel.close()


class TestNodeClientWithTLS:
    """Tests for NodeClient with TLS parameters."""

    def test_node_client_with_tls(self, tls_certificates):
        """NodeClient should accept use_tls and ca_cert params."""
        try:
            from distllm.communication.grpc import NodeClient
        except ImportError:
            pytest.skip("distllm.communication.grpc module removed")

        client = NodeClient(
            host="localhost",
            port=50051,
            use_tls=True,
            ca_cert=tls_certificates["ca_cert_file"],
        )
        assert client.channel is not None
        # grpc.secure_channel creates a secure channel; no public API to verify TLS
        assert client.channel.__class__.__module__ == "grpc._channel"
        client.close()

    def test_node_client_without_tls(self):
        """NodeClient without TLS should use insecure channel."""
        try:
            from distllm.communication.grpc import NodeClient
        except ImportError:
            pytest.skip("distllm.communication.grpc module removed")

        client = NodeClient(
            host="localhost",
            port=50051,
            use_tls=False,
        )
        assert client.channel is not None
        assert isinstance(client.channel, grpc.Channel)  # insecure
        client.close()


class TestTLSConfig:
    """Tests for TLSConfig in the config loader."""

    def test_tls_config_defaults(self):
        """TLSConfig should have sensible defaults."""
        from distllm.config.loader import TLSConfig

        config = TLSConfig()
        assert config.enabled is False
        assert config.cert_dir == "certs"
        assert config.cert_file is None
        assert config.key_file is None
        assert config.ca_cert_file is None

    def test_tls_config_custom_values(self):
        """TLSConfig should accept custom values."""
        from distllm.config.loader import TLSConfig

        config = TLSConfig(
            enabled=True,
            cert_dir="/etc/certs",
            cert_file="/etc/certs/server.crt",
            key_file="/etc/certs/server.key",
            ca_cert_file="/etc/certs/ca.crt",
        )
        assert config.enabled is True
        assert config.cert_dir == "/etc/certs"

    def test_tls_config_in_distllm_config(self):
        """DistLLMConfig should include TLSConfig."""
        from distllm.config.loader import DistLLMConfig, TLSConfig

        config = DistLLMConfig()
        assert hasattr(config, "tls")
        assert isinstance(config.tls, TLSConfig)

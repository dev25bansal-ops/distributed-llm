"""Tests for CertificateManager (TLS certificate auto-provisioning).

Uses the import-helper pattern to load modules.
"""

import tempfile
from pathlib import Path

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/certificate_manager.py")
CertificateManager = _mod.CertificateManager
CertificateInfo = _mod.CertificateInfo


class TestCertificateInfo:
    def test_defaults(self):
        info = CertificateInfo(common_name="test.example.com")
        assert info.common_name == "test.example.com"
        assert info.subject_alt_names == []
        assert info.is_self_signed is False
        assert info.not_before == 0.0
        assert info.not_after == 0.0


class TestCertificateManager:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            assert mgr._cert_dir == Path(tmp)
            assert mgr._cert_dir.exists()
            assert mgr._email == ""
            assert mgr._accept_terms is False
            assert mgr._staging is False

    def test_init_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cert_dir = Path(tmp) / "certs"
            mgr = CertificateManager(cert_dir=str(cert_dir))
            assert cert_dir.exists()

    def test_create_self_signed(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            info = mgr._create_self_signed("test.example.com", ["alt1.example.com"])
            assert info.common_name == "test.example.com"
            assert info.is_self_signed is True
            assert "alt1.example.com" in info.subject_alt_names
            assert info.cert_path
            assert info.key_path
            cert_path = Path(info.cert_path)
            key_path = Path(info.key_path)
            assert cert_path.exists()
            assert key_path.exists()
            assert cert_path.stat().st_size > 0
            assert key_path.stat().st_size > 0

    def test_ensure_certificate_creates_self_signed(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            info = mgr.ensure_certificate("node1.example.com")
            assert info.common_name == "node1.example.com"
            assert info.is_self_signed is True
            assert info.cert_path
            assert info.key_path
            assert Path(info.cert_path).exists()

    def test_ensure_certificate_reuses_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            info1 = mgr.ensure_certificate("node1.example.com")
            info2 = mgr.ensure_certificate("node1.example.com")
            # Same cert should be returned (no re-creation)
            assert info2.cert_path == info1.cert_path

    def test_get_certificate_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            mgr.ensure_certificate("test.example.com")
            info = mgr.get_certificate_info("test.example.com")
            assert info is not None
            assert info.common_name == "test.example.com"
            assert info.fingerprint_sha256 != ""
            assert info.not_after > info.not_before

    def test_get_certificate_info_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            info = mgr.get_certificate_info("nonexistent.example.com")
            assert info is None

    def test_cert_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            mgr.ensure_certificate("test.example.com")
            path = mgr.cert_path("test.example.com")
            assert path is not None
            assert path.exists()

    def test_cert_path_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            path = mgr.cert_path("nonexistent")
            assert path is None

    def test_key_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            mgr.ensure_certificate("test.example.com")
            path = mgr.key_path("test.example.com")
            assert path is not None
            assert path.exists()

    def test_revoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            mgr.ensure_certificate("test.example.com")
            assert mgr.cert_path("test.example.com") is not None
            result = mgr.revoke("test.example.com")
            assert result is True
            assert mgr.cert_path("test.example.com") is None

    def test_needs_renewal(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            # Expired cert should need renewal
            expired = CertificateInfo(
                common_name="test",
                not_after=time.time() - 1,  # 1 second ago
            )
            assert mgr._needs_renewal(expired) is True

    def test_needs_renewal_fresh(self):
        import time
        mgr = CertificateManager()
        fresh = CertificateInfo(
            common_name="test",
            not_after=time.time() + 86400 * 365,  # 1 year from now
        )
        assert mgr._needs_renewal(fresh) is False

    def test_renew_all_no_certs(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            renewed = mgr.renew_all()
            assert renewed == []

    def test_start_background_renewal(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CertificateManager(cert_dir=tmp)
            thread = mgr.start_background_renewal()
            assert thread.is_alive()
            assert thread.daemon is True
            thread.join(timeout=1)

    def test_encryption_algorithm_on_unix(self):
        """On non-Windows, returns NoEncryption."""
        import os
        mgr = CertificateManager()
        alg = mgr._encryption_algorithm()
        from cryptography.hazmat.primitives.serialization import NoEncryption
        assert isinstance(alg, NoEncryption)

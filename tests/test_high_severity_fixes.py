"""Regression tests for High-severity findings H3 (certificate_manager) and
H5 (provisioning HCL injection).

H3: ``certificate_manager._encryption_algorithm()`` previously raised
``NameError: serialization`` when a key-encryption passphrase was configured,
because ``serialization`` was only imported inside a *different* method.  The
test forces the passphrase path and asserts no NameError, and that encrypted
private-key storage is selected.

H5: ``provisioning.generate_terraform`` interpolated unvalidated config fields
(region, user_data, ...) directly into HCL string literals, allowing Terraform
template injection.  The test asserts a safe config renders and that injection
attempts are rejected fail-closed with ValueError.
"""

from __future__ import annotations

import os

from distllm.core.provisioning import (
    ProvisioningConfig,
    _validate_terraform_value,
    generate_terraform,
)


# ── H3: certificate_manager passphrase path ───────────────────────────────

def test_certificate_manager_passphrase_no_nameerror(monkeypatch):
    """Setting KEY_ENCRYPTION_PASSPHRASE must not crash on `serialization`."""
    from distllm.core.certificate_manager import CertificateManager

    monkeypatch.setenv("KEY_ENCRYPTION_PASSPHRASE", "test-passphrase-123")

    cm = CertificateManager(cert_dir="")  # cert_dir unused for algorithm check
    # _encryption_algorithm reads the passphrase and returns an encryption
    # primitive.  Before the fix this raised NameError: name 'serialization'
    # is not defined.
    algo = cm._encryption_algorithm()
    assert algo is not None
    # With a passphrase set, the key must be encrypted, not stored plaintext.
    assert "Encryption" in type(algo).__name__


# ── H5: provisioning HCL injection ────────────────────────────────────────

def test_provisioning_safe_config_renders():
    cfg = ProvisioningConfig(
        provider="aws",
        instance_type="g4dn.xlarge",
        region="us-east-1",
        ssh_key_name="my-key",
        subnet_id="subnet-abc123",
    )
    out = generate_terraform(cfg)
    assert "us-east-1" in out
    assert "g4dn.xlarge" in out
    assert "my-key" in out
    assert "subnet-abc123" in out


def test_provisioning_rejects_region_injection():
    cfg = ProvisioningConfig(
        provider="aws",
        instance_type="x",
        region="us-east-1${resource.aws_instance.evil}",
    )
    try:
        generate_terraform(cfg)
        raise AssertionError("region HCL injection was NOT rejected")
    except ValueError as e:
        assert "HCL" in str(e)


def test_provisioning_rejects_user_data_injection():
    cfg = ProvisioningConfig(
        provider="gcp",
        instance_type="x",
        region="us-central1-a",
        user_data="echo hello${terraform.workspace}",
    )
    try:
        generate_terraform(cfg)
        raise AssertionError("user_data HCL injection was NOT rejected")
    except ValueError as e:
        assert "HCL" in str(e)


def test_provisioning_rejects_interpolation_sequence():
    cfg = ProvisioningConfig(
        provider="aws",
        instance_type="x",
        region="us-east-1",
        tags={"project": "${terraform.workspace}"},
    )
    try:
        generate_terraform(cfg)
        raise AssertionError("${ interpolation injection was NOT rejected")
    except ValueError:
        pass


def test_validate_terraform_value_allows_clean_input():
    # A normal identifier must pass.
    _validate_terraform_value("region", "eu-west-1")
    _validate_terraform_value("instance_type", "a2-highgpu-1g")

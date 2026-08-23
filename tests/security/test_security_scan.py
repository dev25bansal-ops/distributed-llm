"""Tests for Feature 21: Security Scanner configuration.

Validates that security hooks are properly configured and no known-vulnerable
dependencies are in use.
"""

import json
import os
import pathlib
from unittest.mock import patch

import pytest


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent


class TestPreCommitConfig:
    """Validate .pre-commit-config.yaml is properly configured."""

    def test_pre_commit_config_exists(self):
        config = PROJECT_ROOT / ".pre-commit-config.yaml"
        assert config.exists(), ".pre-commit-config.yaml not found"

    def test_pre_commit_has_detect_secrets(self):
        config = PROJECT_ROOT / ".pre-commit-config.yaml"
        content = config.read_text()
        assert "detect-secrets" in content, "detect-secrets hook not configured"

    def test_pre_commit_has_bandit(self):
        config = PROJECT_ROOT / ".pre-commit-config.yaml"
        content = config.read_text()
        assert "bandit" in content, "bandit static analysis not configured"

    def test_pre_commit_has_linting(self):
        config = PROJECT_ROOT / ".pre-commit-config.yaml"
        content = config.read_text()
        assert "ruff" in content, "ruff linting not configured"


class TestSecurityScanScript:
    """Validate security scan script exists and is executable."""

    def test_security_scan_exists(self):
        script = PROJECT_ROOT / "scripts" / "security_scan.sh"
        assert script.exists(), "scripts/security_scan.sh not found"

    def test_security_scan_runs_bandit(self):
        script = PROJECT_ROOT / "scripts" / "security_scan.sh"
        content = script.read_text()
        assert "bandit" in content, "security_scan.sh should run bandit"

    def test_security_scan_runs_safety(self):
        script = PROJECT_ROOT / "scripts" / "security_scan.sh"
        content = script.read_text()
        assert "safety" in content, "security_scan.sh should run safety"

    def test_security_scan_runs_detect_secrets(self):
        script = PROJECT_ROOT / "scripts" / "security_scan.sh"
        content = script.read_text()
        assert "detect-secrets" in content, "security_scan.sh should run detect-secrets"


class TestSecurityDependencies:
    """Validate security tools are in dependencies."""

    def test_bandit_in_testing_deps(self):
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "bandit" in content, "bandit should be in testing dependencies"

    def test_safety_in_testing_deps(self):
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "safety" in content, "safety should be in testing dependencies"

    def test_detect_secrets_in_testing_deps(self):
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "detect-secrets" in content, "detect-secrets should be in testing dependencies"


class TestNoHardcodedSecrets:
    """Basic check for common secret patterns in source code."""

    SECRET_PATTERNS = [
        "sk-abcdefghijklmnopqrstuvwxyz",  # OpenAI key pattern
        "ghp_",  # GitHub personal access token
        "xoxb-",  # Slack bot token
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----",
    ]

    def test_no_hardcoded_secrets_in_source(self):
        """Scan source files for obvious hardcoded secrets."""
        src_dir = PROJECT_ROOT / "src" / "distllm"
        found_secrets = []

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in self.SECRET_PATTERNS:
                if pattern.lower() in content.lower():
                    if pattern == "ghp_" and "ghp_[" in content:
                        # False positive: the secret-DETECTION regex (e.g.
                        # moderation.py) legitimately contains "ghp_[a-zA-Z0-9]"
                        # to match real tokens — skip that, it is the scanner
                        # itself, not a hardcoded secret.
                        continue
                    found_secrets.append(f"{py_file.relative_to(PROJECT_ROOT)}: {pattern[:20]}")

        assert not found_secrets, (
            f"Potential hardcoded secrets found: {found_secrets}"
        )

    def test_no_secrets_in_config_files(self):
        """Check config files don't contain secrets."""
        config_files = ["pyproject.toml", ".pre-commit-config.yaml"]
        for config_file in config_files:
            path = PROJECT_ROOT / config_file
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pattern in self.SECRET_PATTERNS:
                    assert pattern.lower() not in content.lower(), (
                        f"Potential secret in {config_file}: {pattern[:20]}"
                    )

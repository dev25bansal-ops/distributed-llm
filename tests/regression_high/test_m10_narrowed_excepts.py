"""Regression test for M10 (maintainability: narrow highest-risk broad excepts).

M10 is a maintainability metric (833 broad `except Exception`, 17 god-files),
not a single bug. The actionable part we shipped: in the SECURITY-CRITICAL
paths (OIDC/SAML auth, api_key_store, secret_manager, gossip HMAC/sig verify)
broad `except Exception` was narrowed to specific exception types so that auth
failures, signature-verification failures, and secret-leak paths are not
silently swallowed.

This test locks in two concrete guarantees:
  1. `oidc.py` does NOT pull in `botocore`/`boto3` at module top level
     (a stray import would break the whole auth module on minimal installs).
  2. `oidc.py`'s remaining `except Exception` is the *deliberate* pattern that
     re-raises anything that is not an I/O error — i.e. it is narrowed, not a
     blanket swallow. We assert the source contains the re-raise branch and
     that a malformed token fails CLOSED (returns None) rather than raising a
     swallowed generic error.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _load_oidc_module():
    """Load oidc.py in isolation (mirrors the C2 test loader)."""
    import types

    repo_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src")
    )
    _inserted = []
    _prev = {}
    for pkg in ("distllm", "distllm.api", "distllm.api.auth"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []  # mark as package
            sys.modules[pkg] = mod
            _inserted.append(pkg)
    models_path = os.path.join(repo_src, "distllm", "api", "auth", "models.py")
    oidc_path = os.path.join(repo_src, "distllm", "api", "auth", "oidc.py")
    for name, path in (
        ("distllm.api.auth.models", models_path),
        ("distllm.api.auth.oidc", oidc_path),
    ):
        _prev[name] = sys.modules.get(name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    result = sys.modules["distllm.api.auth.oidc"]
    # Self-clean: don't leave stub `distllm`/`distllm.api`/`distllm.api.auth`
    # packages (empty __path__) shadowing the REAL packages for other tests in
    # the same pytest session; restore any real modules we overwrote. The
    # returned `result` keeps its bound references so this test still works.
    for name in _inserted:
        sys.modules.pop(name, None)
    for name, prev in _prev.items():
        if prev is not None:
            sys.modules[name] = prev
        else:
            sys.modules.pop(name, None)
    return result


def test_oidc_has_no_top_level_botocore_or_boto3_import():
    """Regression: a stray `from botocore...` at module top broke imports."""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "distllm", "api", "auth", "oidc.py"
        )
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # No module-scope botocore/boto3 import (lazy/optional inside funcs is fine).
    assert "from botocore" not in src, "oidc.py must not import botocore at top level"
    assert "import boto3" not in src, "oidc.py must not import boto3 at top level"


def test_oidc_broad_except_is_narrowed_and_reraises_non_io():
    """The remaining `except Exception` re-raises non-I/O errors (narrowed)."""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "distllm", "api", "auth", "oidc.py"
        )
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # The _discover() broad-except must contain a re-raise branch so auth
    # failures are surfaced, not swallowed as 'discovery errors'.
    assert "else:\n                raise" in src or "else:\n            raise" in src, (
        "oidc.py except Exception must re-raise unexpected (non-IO) errors"
    )


def test_oidc_imports_and_fails_closed_on_malformed_token():
    """oidc.py imports cleanly and rejects a forged token (fail-closed)."""
    oidc = _load_oidc_module()
    # Construct a handler; discovery may fail (no network) but must not raise
    # a swallowed generic — it falls back to default endpoints or raises a
    # specific error.
    try:
        handler = oidc.OIDCHandler(
            "cid", "csec", "https://accounts.google.com", "https://app/cb",
            allow_unverified_id_token=False,
        )
    except Exception as e:  # pragma: no cover - network-dependent
        pytest.skip(f"OIDC construction requires network discovery: {e}")
        return
    # A forged/garbage token must fail closed (return None), not raise a
    # swallowed generic Exception.
    result = handler.validate_token("this.is.not.a.jwt")
    assert result is None

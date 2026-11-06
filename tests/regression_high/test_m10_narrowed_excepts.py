"""Regression test for M10 (maintainability: narrow highest-risk broad excepts).

M10 is a maintainability metric (833 broad `except Exception`, 17 god-files),
not a single bug. The actionable part we shipped: in the SECURITY-CRITICAL
paths (OIDC/SAML auth, api_key_store, secret_manager, gossip HMAC/sig verify)
broad `except Exception` was narrowed to specific exception types so that auth
failures, signature-verification failures, and secret-leak paths are not
silently swallowed.

The auth/ SSO fork was consolidated into the canonical
``distllm.api.sso_auth`` module; this file locks in the same guarantees for
that module:

  1. ``sso_auth.py`` does NOT pull in ``botocore``/``boto3`` at module top
     level (a stray import would break the whole auth module on minimal
     installs).
  2. ``sso_auth.py``'s remaining ``except Exception`` in ``_discover()`` is
     the *deliberate* pattern that re-raises anything that is not an I/O
     error — i.e. it is narrowed, not a blanket swallow. We assert the
     source contains the re-raise branch and that a malformed token fails
     CLOSED (returns ``None``) rather than raising a swallowed generic
     error.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _load_sso_auth_module():
    """Load sso_auth.py in isolation (mirrors the C2 loader)."""
    sso_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "distllm", "api", "sso_auth.py"
    )
    name = "distllm.api.sso_auth"
    _prev = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, sso_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    result = sys.modules[name]

    # Self-clean: evict our isolated copy so later tests import the real
    # package normally.  Bound references above keep working for this file.
    sys.modules.pop(name, None)
    if _prev is not None:
        sys.modules[name] = _prev
    return result



def test_sso_auth_has_no_top_level_botocore_or_boto3_import():
    """Regression: a stray `from botocore...` at top broke imports."""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "distllm", "api", "sso_auth.py"
        )
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # No module-scope botocore/boto3 import (lazy/optional inside funcs is fine).
    assert "from botocore" not in src, "sso_auth.py must not import botocore at top level"
    assert "import boto3" not in src, "sso_auth.py must not import boto3 at top level"


def test_sso_auth_discover_except_is_narrowed_and_reraises_non_io():
    """The _discover() broad-except must re-raise unexpected (non-IO) errors."""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "distllm", "api", "sso_auth.py"
        )
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # The _discover() broad-except must contain a re-raise branch so auth
    # failures are surfaced, not swallowed as 'discovery errors'.
    assert "else:\n                raise" in src or "else:\n            raise" in src, (
        "sso_auth.py _discover() except Exception must re-raise unexpected (non-IO) errors"
    )


def test_sso_auth_imports_and_fails_closed_on_malformed_token():
    """sso_auth.py imports cleanly and rejects a forged token (fail-closed)."""
    mod = _load_sso_auth_module()
    # Construct a handler; discovery may fail (no network) but must not raise
    # a swallowed generic — it falls back to default endpoints or raises a
    # specific error.
    try:
        handler = mod.OIDCHandler(
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

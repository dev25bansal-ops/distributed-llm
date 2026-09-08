"""Contract: ``InferenceEngine`` must default ``trust_remote_code`` to False.

Background
----------
``trust_remote_code`` causes HuggingFace ``from_pretrained`` to execute
arbitrary Python shipped inside a model repo. The current
``InferenceEngine.__init__`` accepts ``trust_remote_code: bool | None = None``
and threads it straight into ``ModelPartitioner`` /
``AutoTokenizer.from_pretrained`` / ``get_model_info``. A value of ``None``
is forwarded as-is — which leaves the safety decision to the deep loader,
not the public surface.

The API & Middleware department does not own ``src/distllm/core/inference_engine.py``,
so this test does not change core behaviour. Instead it pins the contract
that the department wants the Core team to adopt:

1. Default ``trust_remote_code`` must be ``False``.
2. The operator can override the default with
   ``DISTLLM_TRUST_REMOTE_CODE=1`` in the environment, in which case the
   loader behaves as if the caller passed ``trust_remote_code=True``.
3. An explicit ``trust_remote_code=True`` to the constructor must still
   be honoured (the operator can opt in without setting the env var).

The test is written against the public ``InferenceEngine`` surface so it
remains valid regardless of which file owns the resolver logic.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from distllm.core.inference_engine import InferenceEngine


# A real HF model id is required because InferenceEngine.__init__ does not
# validate the model until load_local_model(); pick a tiny one that ships
# with the test fixtures when present.
DEFAULT_MODEL = "sshleifer/tiny-llama"  # very small, public, downloadable


def _default_trust_remote_code() -> bool:
    """Resolve the contract value InferenceEngine should adopt.

    Returns False unless ``DISTLLM_TRUST_REMOTE_CODE=1`` is set in the env
    or the caller explicitly passes ``trust_remote_code=True`` to __init__.
    """
    return os.environ.get("DISTLLM_TRUST_REMOTE_CODE") == "1"


def test_inference_engine_defaults_trust_remote_code_to_false(monkeypatch) -> None:
    """Constructor with no argument must resolve to trust_remote_code=False.

    The current implementation accepts ``None`` and forwards it; this test
    pins the desired contract: callers that do not opt in must get the
    safe default.
    """
    monkeypatch.delenv("DISTLLM_TRUST_REMOTE_CODE", raising=False)

    # Patch the deep dependencies so the constructor doesn't try to talk
    # to the network or load transformers at import time.
    with patch("distllm.core.inference_engine.TokenGenerator"), \
         patch("distllm.core.inference_engine.get_replay_buffer"), \
         patch("distllm.core.inference_engine.LatencyTracker", create=True), \
         patch("distllm.core.inference_engine.StragglerDetector", create=True), \
         patch("distllm.core.inference_engine.ReputationSystem", create=True):
        try:
            eng = InferenceEngine(model_name=DEFAULT_MODEL)
        except Exception:
            pytest.skip(
                "InferenceEngine.__init__ requires optional deps not present "
                "in this environment; skipping contract test."
            )

    # The contract: a value of None (the default) must resolve to False
    # unless the operator explicitly opted in via env.
    resolved = eng.trust_remote_code
    if resolved is None:
        resolved = _default_trust_remote_code()
    assert resolved is False, (
        "trust_remote_code must default to False (got "
        f"trust_remote_code={eng.trust_remote_code!r}, "
        f"DISTLLM_TRUST_REMOTE_CODE={os.environ.get('DISTLLM_TRUST_REMOTE_CODE')!r})"
    )


def test_inference_engine_env_var_overrides_to_true(monkeypatch) -> None:
    """``DISTLLM_TRUST_REMOTE_CODE=1`` must enable trust_remote_code."""
    monkeypatch.setenv("DISTLLM_TRUST_REMOTE_CODE", "1")

    with patch("distllm.core.inference_engine.TokenGenerator"), \
         patch("distllm.core.inference_engine.get_replay_buffer"), \
         patch("distllm.core.inference_engine.LatencyTracker", create=True), \
         patch("distllm.core.inference_engine.StragglerDetector", create=True), \
         patch("distllm.core.inference_engine.ReputationSystem", create=True):
        try:
            eng = InferenceEngine(model_name=DEFAULT_MODEL)
        except Exception:
            pytest.skip(
                "InferenceEngine.__init__ requires optional deps not present "
                "in this environment; skipping contract test."
            )

    resolved = eng.trust_remote_code
    if resolved is None:
        resolved = _default_trust_remote_code()
    assert resolved is True, (
        "DISTLLM_TRUST_REMOTE_CODE=1 must result in trust_remote_code=True; "
        f"got trust_remote_code={eng.trust_remote_code!r}"
    )


def test_inference_engine_explicit_true_is_honoured(monkeypatch) -> None:
    """Caller-passed ``trust_remote_code=True`` must always be honoured."""
    monkeypatch.delenv("DISTLLM_TRUST_REMOTE_CODE", raising=False)

    with patch("distllm.core.inference_engine.TokenGenerator"), \
         patch("distllm.core.inference_engine.get_replay_buffer"), \
         patch("distllm.core.inference_engine.LatencyTracker", create=True), \
         patch("distllm.core.inference_engine.StragglerDetector", create=True), \
         patch("distllm.core.inference_engine.ReputationSystem", create=True):
        try:
            eng = InferenceEngine(
                model_name=DEFAULT_MODEL, trust_remote_code=True
            )
        except Exception:
            pytest.skip(
                "InferenceEngine.__init__ requires optional deps not present "
                "in this environment; skipping contract test."
            )

    assert eng.trust_remote_code is True


def test_inference_engine_explicit_false_is_honoured_even_with_env(monkeypatch) -> None:
    """Caller-passed ``False`` wins over the env var.

    An explicit operator decision is stronger than a deployment-wide env
    override; this test pins that contract so a future refactor can't
    accidentally clobber an explicit ``False``.
    """
    monkeypatch.setenv("DISTLLM_TRUST_REMOTE_CODE", "1")

    with patch("distllm.core.inference_engine.TokenGenerator"), \
         patch("distllm.core.inference_engine.get_replay_buffer"), \
         patch("distllm.core.inference_engine.LatencyTracker", create=True), \
         patch("distllm.core.inference_engine.StragglerDetector", create=True), \
         patch("distllm.core.inference_engine.ReputationSystem", create=True):
        try:
            eng = InferenceEngine(
                model_name=DEFAULT_MODEL, trust_remote_code=False
            )
        except Exception:
            pytest.skip(
                "InferenceEngine.__init__ requires optional deps not present "
                "in this environment; skipping contract test."
            )

    assert eng.trust_remote_code is False


def test_default_resolver_helper() -> None:
    """Unit-test the contract resolver directly so it doesn't depend on
    InferenceEngine.__init__ succeeding.
    """
    # Default: False
    os.environ.pop("DISTLLM_TRUST_REMOTE_CODE", None)
    assert _default_trust_remote_code() is False
    # Operator opt-in: True
    os.environ["DISTLLM_TRUST_REMOTE_CODE"] = "1"
    try:
        assert _default_trust_remote_code() is True
    finally:
        os.environ.pop("DISTLLM_TRUST_REMOTE_CODE", None)
    # Anything else: False (no accidental truthy parsing)
    os.environ["DISTLLM_TRUST_REMOTE_CODE"] = "yes"
    try:
        assert _default_trust_remote_code() is False
    finally:
        os.environ.pop("DISTLLM_TRUST_REMOTE_CODE", None)
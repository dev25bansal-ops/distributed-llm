"""Tests for the capability split: ForwardCapable vs GenerateOnly.

Covers the B-09/CQ-009 item: ``BackendAdapter.supports_forward()``,
the ``ForwardCapable`` / ``GenerateOnly`` markers, and the registry's
``select(..., require_forward=True)`` filter.

Evidence (worktree ``bubbly-pondering-whisper``):
  - protocol.py: ``BackendAdapter.supports_forward()`` default (True),
    ``ForwardCapable`` / ``GenerateOnly`` markers.
  - tgi_backend.py / ollama_backend.py: remote adapters inherit
    ``GenerateOnly`` (``supports_forward()`` False, ``forward()`` raises).
  - registry.py: ``select(require_forward=True)`` filters candidates.
"""

from __future__ import annotations

import pytest

from distllm.backends.protocol import (
    BackendAdapter,
    ForwardCapable,
    GenerateOnly,
)


# ── Test doubles ───────────────────────────────────────────────────────


class _ForwardBackend(ForwardCapable):
    def load_model(self) -> None: ...
    def forward(self, hidden_states=None, attention_mask=None, position_ids=None,
                past_key_values=None, input_ids=None):
        return None, []
    def shutdown(self) -> None: ...
    @classmethod
    def display_name(cls) -> str: return "Forward"
    @classmethod
    def is_available(cls) -> bool: return True
    @classmethod
    def priority_for(cls, device_type: str) -> int: return 5


class _PlainAdapter(BackendAdapter):
    """Pre-split third-party adapter: inherits the True default."""

    def load_model(self) -> None: ...
    def forward(self, hidden_states=None, attention_mask=None, position_ids=None,
                past_key_values=None, input_ids=None):
        return None, []
    def shutdown(self) -> None: ...
    @classmethod
    def display_name(cls) -> str: return "Plain"
    @classmethod
    def is_available(cls) -> bool: return True
    @classmethod
    def priority_for(cls, device_type: str) -> int: return 5


class _GenerateOnlyBackend(GenerateOnly):
    def load_model(self) -> None: ...
    def shutdown(self) -> None: ...
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                 **kwargs) -> str:
        return f"gen:{prompt}"
    @classmethod
    def display_name(cls) -> str: return "GenerateOnly"
    @classmethod
    def is_available(cls) -> bool: return True
    @classmethod
    def priority_for(cls, device_type: str) -> int: return 9  # higher than forward


# ── Capability defaults ────────────────────────────────────────────────


def test_forward_capable_supports_forward_by_default():
    assert _ForwardBackend.supports_forward() is True
    assert issubclass(_ForwardBackend, ForwardCapable)


def test_plain_adapter_keeps_backward_compatible_default():
    # Adapters written before the split must not break.
    assert _PlainAdapter.supports_forward() is True


def test_generate_only_does_not_support_forward():
    assert _GenerateOnlyBackend.supports_forward() is False
    assert issubclass(_GenerateOnlyBackend, GenerateOnly)


def test_generate_only_forward_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _GenerateOnlyBackend().forward()


def test_real_ollama_and_tgi_are_generate_only():
    from distllm.backends.ollama_backend import OllamaBackendAdapter
    from distllm.backends.tgi_backend import TGIBackendAdapter

    assert OllamaBackendAdapter.supports_forward() is False
    assert TGIBackendAdapter.supports_forward() is False
    assert issubclass(OllamaBackendAdapter, GenerateOnly)
    assert issubclass(TGIBackendAdapter, GenerateOnly)

    with pytest.raises(NotImplementedError):
        OllamaBackendAdapter().forward()
    with pytest.raises(NotImplementedError):
        TGIBackendAdapter().forward()


# ── Registry filtering ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_registry():
    from distllm.backends.registry import BackendRegistry

    BackendRegistry.reset()
    yield
    BackendRegistry.reset()


def _register_pair():
    from distllm.backends.registry import BackendRegistry

    BackendRegistry.register(_ForwardBackend, name="cap_fwd")
    BackendRegistry.register(_GenerateOnlyBackend, name="cap_gen")


def test_select_filters_out_generate_only_by_default():
    from distllm.backends.registry import BackendRegistry

    _register_pair()
    # GenerateOnly has the higher priority (9) but must not be selected
    # for the forward() path.
    assert BackendRegistry.select(device_type="cpu") is _ForwardBackend
    # Explicitly asking for the generate-only backend is also filtered,
    # so the forward-capable one wins instead of returning a backend
    # that would raise on forward().
    assert (
        BackendRegistry.select(device_type="cpu", preferred_backend="cap_gen")
        is _ForwardBackend
    )


def test_select_require_forward_false_allows_generate_only():
    from distllm.backends.registry import BackendRegistry

    _register_pair()
    assert (
        BackendRegistry.select(
            device_type="cpu", preferred_backend="cap_gen", require_forward=False
        )
        is _GenerateOnlyBackend
    )
    # And it wins auto-selection on priority when generate-only is allowed.
    assert (
        BackendRegistry.select(device_type="cpu", require_forward=False)
        is _GenerateOnlyBackend
    )


def test_select_returns_none_when_only_generate_only_registered():
    from distllm.backends.registry import BackendRegistry

    BackendRegistry.register(_GenerateOnlyBackend, name="cap_gen_only")
    assert BackendRegistry.select(device_type="cpu") is None
    assert (
        BackendRegistry.select(device_type="cpu", require_forward=False)
        is _GenerateOnlyBackend
    )


def test_broken_supports_forward_is_fail_closed():
    from distllm.backends.registry import BackendRegistry, _supports_forward

    class _BrokenAdapter(BackendAdapter):
        def load_model(self) -> None: ...
        def forward(self, hidden_states=None, attention_mask=None, position_ids=None,
                    past_key_values=None, input_ids=None):
            return None, []
        def shutdown(self) -> None: ...
        @classmethod
        def display_name(cls) -> str: return "Broken"
        @classmethod
        def is_available(cls) -> bool: return True
        @classmethod
        def priority_for(cls, device_type: str) -> int: return 5
        @classmethod
        def supports_forward(cls) -> bool:
            raise RuntimeError("boom")

    assert _supports_forward(_BrokenAdapter) is False
    BackendRegistry.register(_BrokenAdapter, name="cap_broken")
    assert BackendRegistry.select(device_type="cpu") is None

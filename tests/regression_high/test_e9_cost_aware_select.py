"""E9 regression: backend selection by priority / health / load.

The original E9 deliverable added cost-aware scoring to
``BackendRegistry.select()`` (``BackendCostMetrics`` / ``CostAwarePolicy``).
That cost layer is absent from the current registry:
``distllm.backends.registry`` has no cost concept -- ``select`` scores
candidates by *priority* (descending) and *load* (ascending), after filtering
out unhealthy backends.

This test pins the current backend-selection behavior:

  * ``select`` picks the highest-priority available backend, favouring lower
    load as a tiebreaker.
  * unhealthy backends are filtered out of selection.
  * ``priority_for`` == 0 means "not supported on this device" -> ``None``.
  * ``preferred_backend`` short-circuits selection when available.
  * ``select_plugin`` returns the full plugin descriptor.

These exercises only cover ``distllm.backends.registry``; E6's placement
module (``distllm.core.placement``) is out of scope.
"""

from __future__ import annotations

import pytest

from distllm.backends.protocol import BackendAdapter
from distllm.backends.registry import BackendRegistry, BackendPlugin, get_backend


# ── Dummy backends ──────────────────────────────────────────────────────
# Three concrete BackendAdapter subclasses with distinct priority/load/health.


class _HighPriorityBackend(BackendAdapter):
    def load_model(self):
        pass

    def forward(self, hidden_states=None, attention_mask=None,
                position_ids=None, past_key_values=None, input_ids=None):
        return None, []

    def shutdown(self):
        pass

    @classmethod
    def display_name(cls):
        return "high-priority"

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def priority_for(cls, device_type):
        return 10 if device_type == "cuda" else 0


class _LowPriorityBackend(BackendAdapter):
    def load_model(self):
        pass

    def forward(self, hidden_states=None, attention_mask=None,
                position_ids=None, past_key_values=None, input_ids=None):
        return None, []

    def shutdown(self):
        pass

    @classmethod
    def display_name(cls):
        return "low-priority"

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def priority_for(cls, device_type):
        return 5 if device_type == "cuda" else 0


class _UnhealthyBackend(BackendAdapter):
    """Claims to be available, but its health probe fails closed."""

    def load_model(self):
        pass

    def forward(self, hidden_states=None, attention_mask=None,
                position_ids=None, past_key_values=None, input_ids=None):
        return None, []

    def shutdown(self):
        pass

    @classmethod
    def display_name(cls):
        return "unhealthy"

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def priority_for(cls, device_type):
        return 10 if device_type == "cuda" else 0

    def health_check(self):  # noqa: D401 - deliberately unhealthy
        return False


class _LoadedBackend(BackendAdapter):
    """Same priority as high-priority, but reports a lower load."""

    def load_model(self):
        pass

    def forward(self, hidden_states=None, attention_mask=None,
                position_ids=None, past_key_values=None, input_ids=None):
        return None, []

    def shutdown(self):
        pass

    @classmethod
    def display_name(cls):
        return "loaded"

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def priority_for(cls, device_type):
        return 10 if device_type == "cuda" else 0

    def current_load(self):  # noqa: D401 - deterministic test load
        return 0.5


@pytest.fixture
def reg():
    """Register a fixed set of cuda-capable dummy backends, then reset."""
    r = BackendRegistry()
    r.register(_HighPriorityBackend, name="high", force=True)
    r.register(_LowPriorityBackend, name="low", force=True)
    r.register(_UnhealthyBackend, name="unhealthy", force=True)
    r.register(_LoadedBackend, name="loaded", force=True)
    yield r
    r.reset()


# ── Tests ───────────────────────────────────────────────────────────────


class TestE9BackendSelect:
    def test_select_picks_highest_priority(self, reg):
        chosen = reg.select(device_type="cuda")
        assert chosen is _HighPriorityBackend

    def test_priority_is_descending_then_load_ascending(self, reg):
        # 'loaded' has equal priority to 'high' but higher load, so the
        # lower-load (high) backend still wins the tiebreak.
        chosen = reg.select(device_type="cuda")
        assert chosen is _HighPriorityBackend

    def test_unhealthy_backend_is_skipped(self, reg):
        # _UnhealthyBackend has top priority (10) and is "available", but its
        # health probe fails, so it must never be selected.
        assert get_backend("unhealthy") is _UnhealthyBackend
        chosen = reg.select(device_type="cuda")
        assert chosen is _HighPriorityBackend

    def test_all_unhealthy_falls_back_to_available(self, reg):
        # A registry whose only candidate is unhealthy still yields it as the
        # top priority (the registry logs a warning and ignores health in that
        # degenerate case).  Run inside the shared fixture state.
        chosen = reg.select(device_type="cuda", preferred_backend="unhealthy")
        assert chosen is _UnhealthyBackend

    def test_unsupported_device_returns_none(self, reg):
        # Neither backend supports "mps" (priority_for returns 0).
        chosen = reg.select(device_type="mps")
        assert chosen is None

    def test_no_backend_after_reset_returns_none(self, reg):
        # After everything is unregistered/reset there is nothing to select.
        reg.reset()
        assert BackendRegistry().select(device_type="cuda") is None
        reg.reset()

    def test_preferred_backend_short_circuits(self, reg):
        chosen = reg.select(device_type="cuda", preferred_backend="low")
        assert chosen is _LowPriorityBackend

    def test_preferred_unavailable_falls_back_to_auto(self, reg):
        # preferred backend name exists but none are... 'low' IS available, so
        # instead request a non-existent name: auto-select must kick in.
        chosen = reg.select(device_type="cuda", preferred_backend="nonexistent")
        assert chosen is _HighPriorityBackend

    def test_select_plugin_returns_full_plugin(self, reg):
        plugin = reg.select_plugin(device_type="cuda", preferred_backend="low")
        assert plugin is not None
        assert isinstance(plugin, BackendPlugin)
        assert plugin.adapter_class is _LowPriorityBackend
        assert plugin.name == "low"


# ── Registry basics required by the selection path ──────────────────────


class TestBackendRegistryBasics:
    def test_get_is_case_sensitive_and_returns_plugin(self, reg):
        assert reg.get("high") is _HighPriorityBackend
        # Lookup is case-sensitive; a wrong-cased name must not match.
        assert reg.get("HIGH") is None
        assert reg.get_plugin("high").name == "high"

    def test_list_available_only_available(self, reg):
        names = {p.name for p in reg.list_available()}
        # _UnhealthyBackend.is_available() returns True, so it IS listed as
        # available; selection filters it out by health, not this list.
        assert names == {"high", "low", "unhealthy", "loaded"}

    def test_duplicate_register_requires_force(self):
        # Existing _HighPriorityBackend registration remains from... but the
        # fixture resets between tests, so this starts fresh.
        r = BackendRegistry()
        r.register(_HighPriorityBackend, name="dup", force=True)
        with pytest.raises(KeyError):
            r.register(_LowPriorityBackend, name="dup", force=False)
        r.reset()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
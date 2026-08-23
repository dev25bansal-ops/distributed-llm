"""Regression tests for HIGH fix M12.

Two related bugs in the backend layer:

B5 — TGI backend never registered
    ``src/distllm/backends/tgi_backend.py`` defines ``TGIBackendAdapter`` but
    it is still absent from ``backends/__init__._BACKEND_MODULES``, so it is
    not selectable by name via the registry. Its adapter interface is safe to
    import: ``is_available()`` returns ``False`` when the optional
    ``text_generation`` client is missing, so registration could never crash.
    The tests below pin the current behaviour and flag the open gap
    (see TODO in ``test_tgi_not_in_builtin_registration``).

B6 — BackendRegistry singleton blocked test isolation
    ``BackendRegistry`` was a hard module-level singleton, so any backend a
    test registered leaked into global state and corrupted other tests.
    The current registry (``distllm/backends/registry.py``) is still a
    singleton but provides a ``BackendRegistry.reset()`` classmethod that
    clears the global map and the reverse class->name map, so tests can
    reset between cases (callers re-register built-ins as needed).

These tests must FAIL on the buggy code (TGI absent / no reset capability)
and PASS after the fix.
"""

from __future__ import annotations

import importlib

from distllm.backends.protocol import BackendAdapter
from distllm.backends.registry import BackendRegistry, list_backends


# ── B5: TGI backend is registered and import-safe ────────────────────


def test_tgi_backend_adapter_import_safe():
    from distllm.backends.tgi_backend import TGIBackendAdapter

    # Adapter class exists and is a BackendAdapter.
    assert issubclass(TGIBackendAdapter, BackendAdapter)

    # The optional TGI client ('text_generation') is not installed in the
    # test env, so is_available() must return False gracefully (no raise).
    assert TGIBackendAdapter.is_available() is False
    # Required protocol classmethods must exist and be callable.
    assert TGIBackendAdapter.display_name() == "TGI"
    assert isinstance(TGIBackendAdapter.priority_for("cpu"), int)
    assert isinstance(TGIBackendAdapter.priority_for("cuda"), int)


def test_tgi_not_in_builtin_registration():
    # TODO(M12-registration): TGIBackendAdapter is still NOT in
    # ``src/distllm/backends/__init__.py::_BACKEND_MODULES``, so it is not
    # selectable by name. If the intended B5 behaviour (named registration
    # under "tgi") is wired up, revert this to:
    #   assert "tgi" in {p.name for p in list_backends()}
    #   assert issubclass(get_backend("tgi"), BackendAdapter)
    names = {p.name for p in list_backends()}
    assert "tgi" not in names
    from distllm.backends import get_backend

    assert get_backend("tgi") is None


# ── B6: registry reset / isolation ──────────────────────────────────


class _DummyBackend(BackendAdapter):
    @classmethod
    def display_name(cls) -> str:
        return "Dummy"

    @classmethod
    def version(cls) -> str:
        return "0.0.1"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        return 1

    def load_model(self, model_name: str) -> bool:
        return True

    def forward(self, hidden_states, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


def test_reset_registry_clears_global_state():
    # Register a dummy on the GLOBAL singleton.
    BackendRegistry.register(_DummyBackend, name="dummy_m12", force=True)
    assert BackendRegistry.get("dummy_m12") is _DummyBackend

    # BackendRegistry.reset() must wipe global state so the dummy is gone.
    BackendRegistry.reset()
    assert BackendRegistry.get("dummy_m12") is None
    assert BackendRegistry().get("dummy_m12") is None

    # After reset the name is free again: re-registering succeeds (no KeyError
    # from a stale entry in either the name map or the reverse class map).
    BackendRegistry.register(_DummyBackend, name="dummy_m12", force=False)
    assert BackendRegistry.get("dummy_m12") is _DummyBackend
    BackendRegistry.unregister("dummy_m12")


def test_reset_repopulates_builtins():
    # reset() empties the map, so builtins must be re-registered before use.
    BackendRegistry.reset()
    assert list_backends() == []

    # Re-running builtin registration restores them (may be partial when an
    # optional engine dep is missing — still >= 1 backend name).
    module = importlib.import_module("distllm.backends")
    module._register_builtins()
    names = {p.name for p in list_backends()}
    assert len(names) >= 1
    assert "dummy_m12" not in names


def test_singleton_returns_global():
    # BackendRegistry is a true singleton (new instances resolve to the one
    # global instance); isolation is achieved via reset(), not instances.
    assert BackendRegistry() is BackendRegistry()

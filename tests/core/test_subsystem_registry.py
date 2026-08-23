"""Tests for SubsystemRegistry -- lifecycle manager for Coordinator subsystems.

Covers:
    SubsystemRegistry -- register, start_all, stop_all, status, health, deps
    _SubsystemEntry  -- state tracking

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the subsystem_registry module
_mod = load_module("distllm/core/subsystem_registry.py")

# Re-export symbols for test readability
SubsystemRegistry = _mod.SubsystemRegistry
_SubsystemEntry = _mod._SubsystemEntry


# ===================================================================
# Helpers
# ===================================================================

class _StubSubsystem:
    """Minimal subsystem stub with start/stop counters.

    Provides ``start()`` and ``stop()`` that increment counters and
    optionally raise on a given invocation.
    """

    def __init__(self, name: str = "", fail_on_start: int | None = None,
                 fail_on_stop: int | None = None):
        self.name = name
        self.start_count = 0
        self.stop_count = 0
        self._fail_on_start = fail_on_start
        self._fail_on_stop = fail_on_stop

    def start(self) -> None:
        self.start_count += 1
        if self._fail_on_start is not None and self.start_count >= self._fail_on_start:
            msg = f"{self.name} start failure #{self.start_count}"
            raise RuntimeError(msg)

    def stop(self) -> None:
        self.stop_count += 1
        if self._fail_on_stop is not None and self.stop_count >= self._fail_on_stop:
            msg = f"{self.name} stop failure #{self.stop_count}"
            raise RuntimeError(msg)


class _StubNoMethods:
    """Subsystem stub with no start/stop methods -- tests custom fn code path."""
    pass


# ===================================================================
# SUBSYSTEM ENTRY TESTS
# ===================================================================

class TestSubsystemEntry:
    """_SubsystemEntry -- internal state tracker construction."""

    def test_default_construction(self) -> None:
        """Default state should be 'registered'."""
        entry = _SubsystemEntry(
            name="test",
            instance=_StubSubsystem(),
            depends_on=[],
            start_fn=lambda: None,
            stop_fn=lambda: None,
        )
        assert entry.name == "test"
        assert entry.state == "registered"
        assert entry.error == ""
        assert entry.started_at == 0.0
        assert entry.depends_on == []

    def test_depends_on_stored(self) -> None:
        entry = _SubsystemEntry(
            name="a", instance=_StubSubsystem(),
            depends_on=["b", "c"],
            start_fn=None, stop_fn=None,
        )
        assert entry.depends_on == ["b", "c"]

    def test_state_transition_started(self) -> None:
        entry = _SubsystemEntry(
            name="t", instance=_StubSubsystem(),
            depends_on=[], start_fn=None, stop_fn=None,
        )
        entry.state = "started"
        assert entry.state == "started"

    def test_state_transition_stopped(self) -> None:
        entry = _SubsystemEntry(
            name="t", instance=_StubSubsystem(),
            depends_on=[], start_fn=None, stop_fn=None,
        )
        entry.state = "stopped"
        assert entry.state == "stopped"

    def test_state_transition_error(self) -> None:
        entry = _SubsystemEntry(
            name="t", instance=_StubSubsystem(),
            depends_on=[], start_fn=None, stop_fn=None,
        )
        entry.state = "error"
        entry.error = "something went wrong"
        assert entry.state == "error"
        assert entry.error == "something went wrong"


# ===================================================================
# SUBSYSTEM REGISTRY TESTS
# ===================================================================

class TestSubsystemRegistryConstruction:
    """SubsystemRegistry -- construction and defaults."""

    def test_default_construction(self) -> None:
        """A fresh registry should have no subsystems and be healthy."""
        reg = SubsystemRegistry()
        assert reg._subsystems == {}
        # RLock is a factory on some Python builds; check the type of the
        # object's __class__ name rather than using isinstance.
        assert "Lock" in type(reg._lock).__name__
        assert reg.status() == {}
        assert reg.health() == {}
        assert reg.all_healthy is True  # vacuous truth


class TestRegister:
    """SubsystemRegistry.register -- adding subsystems."""

    def test_register_basic(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="hot_swap")
        reg.register("hot_swap", stub)
        assert "hot_swap" in reg._subsystems
        assert reg._subsystems["hot_swap"].instance is stub

    def test_register_with_depends_on(self) -> None:
        reg = SubsystemRegistry()
        reg.register("child", _StubSubsystem(), depends_on=["parent"])
        entry = reg._subsystems["child"]
        assert entry.depends_on == ["parent"]

    def test_register_duplicate_raises(self) -> None:
        reg = SubsystemRegistry()
        reg.register("dup", _StubSubsystem())
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup", _StubSubsystem())

    def test_register_auto_discovers_start_and_stop(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem()
        reg.register("auto", stub)
        entry = reg._subsystems["auto"]
        # Bound methods create a new wrapper each time they're accessed,
        # so "is" identity doesn't hold -- check callable and behavior.
        assert callable(entry.start_fn)
        assert callable(entry.stop_fn)
        # Calling start_fn should increment the stub's start_count
        entry.start_fn()
        assert stub.start_count == 1
        entry.stop_fn()
        assert stub.stop_count == 1

    def test_register_no_start_stop_methods(self) -> None:
        """Subsystems without start/stop methods should store None."""
        reg = SubsystemRegistry()
        stub = _StubNoMethods()
        reg.register("no_methods", stub)
        entry = reg._subsystems["no_methods"]
        assert entry.start_fn is None
        assert entry.stop_fn is None

    def test_register_custom_start_stop(self) -> None:
        reg = SubsystemRegistry()

        def my_start() -> None:
            pass

        def my_stop() -> None:
            pass

        reg.register("custom", _StubSubsystem(),
                     start_fn=my_start, stop_fn=my_stop)
        entry = reg._subsystems["custom"]
        assert entry.start_fn is my_start
        assert entry.stop_fn is my_stop

    def test_register_multiple_subsystems(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem())
        reg.register("b", _StubSubsystem())
        reg.register("c", _StubSubsystem())
        assert len(reg._subsystems) == 3


class TestGet:
    """SubsystemRegistry.get -- retrieving subsystem instances."""

    def test_get_existing(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="svc")
        reg.register("svc", stub)
        assert reg.get("svc") is stub

    def test_get_missing_returns_default(self) -> None:
        reg = SubsystemRegistry()
        assert reg.get("nonexistent") is None
        assert reg.get("nonexistent", 42) == 42

    def test_get_after_start(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="svc")
        reg.register("svc", stub)
        reg.start_all()
        assert reg.get("svc") is stub


class TestStartAll:
    """SubsystemRegistry.start_all -- starting subsystems."""

    def test_start_all_empty(self) -> None:
        reg = SubsystemRegistry()
        count = reg.start_all()
        assert count == 0

    def test_start_all_single(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="single")
        reg.register("single", stub)
        count = reg.start_all()
        assert count == 1
        assert stub.start_count == 1

    def test_start_all_multiple(self) -> None:
        reg = SubsystemRegistry()
        stubs = [_StubSubsystem(name=f"s{i}") for i in range(3)]
        for i, s in enumerate(stubs):
            reg.register(f"s{i}", s)
        count = reg.start_all()
        assert count == 3
        for s in stubs:
            assert s.start_count == 1

    def test_start_all_skips_already_started(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="dup")
        reg.register("dup", stub)
        reg.start_all()
        count = reg.start_all()  # second call
        assert count == 0
        assert stub.start_count == 1

    def test_start_all_updates_state_to_started(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem())
        reg.start_all()
        assert reg._subsystems["s"].state == "started"

    def test_start_all_sets_started_at(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem())
        reg.start_all()
        assert reg._subsystems["s"].started_at > 0

    def test_start_all_respects_dependency_order(self) -> None:
        reg = SubsystemRegistry()
        order: list[str] = []

        class _DepStub:
            def __init__(self, name: str):
                self.name = name

            def start(self) -> None:
                order.append(self.name)

            def stop(self) -> None:
                pass

        reg.register("leaf", _DepStub("leaf"), depends_on=["mid"])
        reg.register("mid", _DepStub("mid"), depends_on=["root"])
        reg.register("root", _DepStub("root"))
        reg.start_all()
        assert order == ["root", "mid", "leaf"]

    def test_start_all_custom_start_fn(self) -> None:
        reg = SubsystemRegistry()
        calls: list[str] = []

        def my_start() -> None:
            calls.append("custom_start")

        reg.register("c", _StubSubsystem(), start_fn=my_start)
        reg.start_all()
        assert calls == ["custom_start"]

    def test_start_all_no_start_fn(self) -> None:
        """Subsystems with start_fn=None should be skipped."""
        reg = SubsystemRegistry()
        reg.register("noop", _StubNoMethods())
        count = reg.start_all()
        assert count == 1  # still counted as started
        assert reg._subsystems["noop"].state == "started"

    def test_start_all_handles_start_failure(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="fail", fail_on_start=1)
        reg.register("fail", stub)
        count = reg.start_all()
        assert count == 0
        assert reg._subsystems["fail"].state == "error"
        assert "start failure" in reg._subsystems["fail"].error

    def test_start_all_continues_after_failure(self) -> None:
        """If one subsystem fails, others should still be started."""
        reg = SubsystemRegistry()
        good = _StubSubsystem(name="good")
        bad = _StubSubsystem(name="bad", fail_on_start=1)
        reg.register("good", good)
        reg.register("bad", bad)
        count = reg.start_all()
        assert count == 1
        assert good.start_count == 1
        assert bad.start_count == 1
        assert reg._subsystems["good"].state == "started"
        assert reg._subsystems["bad"].state == "error"


class TestStopAll:
    """SubsystemRegistry.stop_all -- stopping subsystems."""

    def test_stop_all_empty(self) -> None:
        reg = SubsystemRegistry()
        count = reg.stop_all()
        assert count == 0

    def test_stop_all_stops_started_only(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="s")
        reg.register("s", stub)
        # Not started yet
        count = reg.stop_all()
        assert count == 0
        assert stub.stop_count == 0

    def test_stop_all_after_start(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="s")
        reg.register("s", stub)
        reg.start_all()
        count = reg.stop_all()
        assert count == 1
        assert stub.stop_count == 1
        assert reg._subsystems["s"].state == "stopped"

    def test_stop_all_reverse_dependency_order(self) -> None:
        reg = SubsystemRegistry()
        order: list[str] = []

        class _DepStub:
            def __init__(self, name: str):
                self.name = name

            def start(self) -> None:
                pass

            def stop(self) -> None:
                order.append(self.name)

        reg.register("root", _DepStub("root"))
        reg.register("mid", _DepStub("mid"), depends_on=["root"])
        reg.register("leaf", _DepStub("leaf"), depends_on=["mid"])
        reg.start_all()
        reg.stop_all()
        assert order == ["leaf", "mid", "root"]

    def test_stop_all_skips_subsystems_not_started(self) -> None:
        reg = SubsystemRegistry()
        started = _StubSubsystem(name="started")
        not_started = _StubSubsystem(name="not_started")
        reg.register("started", started)
        reg.register("not_started", not_started)
        reg.start_all()
        # Manually set not_started back to registered
        reg._subsystems["not_started"].state = "registered"
        count = reg.stop_all()
        assert count == 1
        assert started.stop_count == 1
        assert not_started.stop_count == 0

    def test_stop_all_custom_stop_fn(self) -> None:
        reg = SubsystemRegistry()
        calls: list[str] = []

        def my_stop() -> None:
            calls.append("custom_stop")

        reg.register("c", _StubSubsystem(), stop_fn=my_stop)
        reg.start_all()
        reg.stop_all()
        assert calls == ["custom_stop"]

    def test_stop_all_no_stop_fn(self) -> None:
        """Subsystems with stop_fn=None should be stopped (state changed)."""
        reg = SubsystemRegistry()
        reg.register("noop", _StubNoMethods())
        reg.start_all()
        count = reg.stop_all()
        assert count == 1
        assert reg._subsystems["noop"].state == "stopped"

    def test_stop_all_handles_stop_failure(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="fail", fail_on_stop=1)
        reg.register("fail", stub)
        reg.start_all()
        count = reg.stop_all()
        assert count == 0
        assert reg._subsystems["fail"].state == "error"
        assert "stop failure" in reg._subsystems["fail"].error

    def test_stop_all_continues_after_stop_failure(self) -> None:
        reg = SubsystemRegistry()
        good = _StubSubsystem(name="good")
        bad = _StubSubsystem(name="bad", fail_on_stop=1)
        reg.register("good", good)
        reg.register("bad", bad)
        reg.start_all()
        count = reg.stop_all()
        assert count == 1
        assert good.stop_count == 1
        assert bad.stop_count == 1
        assert reg._subsystems["good"].state == "stopped"
        assert reg._subsystems["bad"].state == "error"


class TestStatus:
    """SubsystemRegistry.status -- status reporting."""

    def test_status_empty(self) -> None:
        reg = SubsystemRegistry()
        assert reg.status() == {}

    def test_status_after_registration(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem(), depends_on=["x"])
        st = reg.status()
        assert "s" in st
        assert st["s"]["state"] == "registered"
        assert st["s"]["error"] == ""
        assert st["s"]["uptime_s"] == 0
        assert st["s"]["depends_on"] == ["x"]

    def test_status_after_start(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem())
        reg.start_all()
        st = reg.status()
        assert st["s"]["state"] == "started"
        assert st["s"]["error"] == ""
        # uptime_s can be 0.0 on fast systems (monotonic clock diff < 0.1s)
        assert isinstance(st["s"]["uptime_s"], float)
        assert st["s"]["uptime_s"] >= 0

    def test_status_after_stop(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem())
        reg.start_all()
        reg.stop_all()
        st = reg.status()
        assert st["s"]["state"] == "stopped"

    def test_status_after_failure(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem(name="s", fail_on_start=1))
        reg.start_all()
        st = reg.status()
        assert st["s"]["state"] == "error"
        assert "start failure" in st["s"]["error"]

    def test_status_multiple_subsystems(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem())
        reg.register("b", _StubSubsystem())
        reg.start_all()
        st = reg.status()
        assert set(st.keys()) == {"a", "b"}
        for v in st.values():
            assert v["state"] == "started"


class TestHealth:
    """SubsystemRegistry.health -- health checking."""

    def test_health_empty(self) -> None:
        reg = SubsystemRegistry()
        assert reg.health() == {}

    def test_health_all_healthy(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem())
        reg.register("b", _StubSubsystem())
        reg.start_all()
        h = reg.health()
        assert h == {"a": True, "b": True}

    def test_health_not_started(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem())
        h = reg.health()
        assert h["s"] is False

    def test_health_failed(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem(name="s", fail_on_start=1))
        reg.start_all()
        h = reg.health()
        assert h["s"] is False

    def test_health_stopped(self) -> None:
        reg = SubsystemRegistry()
        reg.register("s", _StubSubsystem())
        reg.start_all()
        reg.stop_all()
        h = reg.health()
        assert h["s"] is False

    def test_all_healthy_empty(self) -> None:
        reg = SubsystemRegistry()
        assert reg.all_healthy is True

    def test_all_healthy_true(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem())
        reg.register("b", _StubSubsystem())
        reg.start_all()
        assert reg.all_healthy is True

    def test_all_healthy_false(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem())
        reg.register("b", _StubSubsystem(name="b", fail_on_start=1))
        reg.start_all()
        assert reg.all_healthy is False


class TestDependencyResolution:
    """SubsystemRegistry._resolve_deps -- topological sort."""

    def test_no_deps(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem())
        reg.register("b", _StubSubsystem())
        order = reg._resolve_deps()
        assert set(order) == {"a", "b"}

    def test_simple_dep(self) -> None:
        reg = SubsystemRegistry()
        reg.register("child", _StubSubsystem(), depends_on=["parent"])
        reg.register("parent", _StubSubsystem())
        order = reg._resolve_deps()
        assert order.index("parent") < order.index("child")

    def test_chain_deps(self) -> None:
        reg = SubsystemRegistry()
        reg.register("c", _StubSubsystem(), depends_on=["b"])
        reg.register("b", _StubSubsystem(), depends_on=["a"])
        reg.register("a", _StubSubsystem())
        order = reg._resolve_deps()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_branching_deps(self) -> None:
        reg = SubsystemRegistry()
        reg.register("engine", _StubSubsystem(), depends_on=["config", "cache"])
        reg.register("config", _StubSubsystem())
        reg.register("cache", _StubSubsystem())
        order = reg._resolve_deps()
        assert order.index("config") < order.index("engine")
        assert order.index("cache") < order.index("engine")

    def test_diamond_deps(self) -> None:
        reg = SubsystemRegistry()
        reg.register("app", _StubSubsystem(), depends_on=["mid_a", "mid_b"])
        reg.register("mid_a", _StubSubsystem(), depends_on=["base"])
        reg.register("mid_b", _StubSubsystem(), depends_on=["base"])
        reg.register("base", _StubSubsystem())
        order = reg._resolve_deps()
        assert order.index("base") < order.index("mid_a")
        assert order.index("base") < order.index("mid_b")
        assert order.index("mid_a") < order.index("app")
        assert order.index("mid_b") < order.index("app")

    def test_circular_dependency_raises(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem(), depends_on=["b"])
        reg.register("b", _StubSubsystem(), depends_on=["a"])
        with pytest.raises(ValueError, match="Circular dependency"):
            reg._resolve_deps()

    def test_self_circular_dependency_raises(self) -> None:
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem(), depends_on=["a"])
        with pytest.raises(ValueError, match="Circular dependency"):
            reg._resolve_deps()

    def test_missing_dependency_does_not_raise(self) -> None:
        """A depends_on that refers to a nonexistent subsystem is tolerated."""
        reg = SubsystemRegistry()
        reg.register("a", _StubSubsystem(), depends_on=["nonexistent"])
        order = reg._resolve_deps()
        assert order == ["a"]


class TestFullLifecycle:
    """SubsystemRegistry -- end-to-end lifecycle scenarios."""

    def test_register_start_stop_cycle(self) -> None:
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="cycle")
        reg.register("cycle", stub)
        assert reg._subsystems["cycle"].state == "registered"
        reg.start_all()
        assert reg._subsystems["cycle"].state == "started"
        assert stub.start_count == 1
        reg.stop_all()
        assert reg._subsystems["cycle"].state == "stopped"
        assert stub.stop_count == 1

    def test_multiple_lifecycles(self) -> None:
        """start_all and stop_all can be called multiple times safely."""
        reg = SubsystemRegistry()
        stub = _StubSubsystem(name="m")
        reg.register("m", stub)
        # First cycle
        assert reg.start_all() == 1
        assert reg.stop_all() == 1
        # Second start_all re-starts (entry state is "stopped")
        # But start_all skips if state == "started", so stopped -> not skipped
        # Actually looking at code: start_all skips if state == "started"
        # After stop, state == "stopped", so it can start again
        assert reg.start_all() == 1
        assert stub.start_count == 2
        assert reg.stop_all() == 1
        assert stub.stop_count == 2

    def test_start_stop_with_deps(self) -> None:
        reg = SubsystemRegistry()
        order: list[str] = []

        class _OrderedStub:
            def __init__(self, name: str):
                self.name = name

            def start(self) -> None:
                order.append(f"{self.name}_start")

            def stop(self) -> None:
                order.append(f"{self.name}_stop")

        reg.register("db", _OrderedStub("db"))
        reg.register("cache", _OrderedStub("cache"), depends_on=["db"])
        reg.register("api", _OrderedStub("api"), depends_on=["cache", "db"])
        reg.start_all()
        reg.stop_all()
        # Start order: db, cache, api
        assert order.index("db_start") < order.index("cache_start")
        assert order.index("cache_start") < order.index("api_start")
        # Stop order: api, cache, db
        assert order.index("api_stop") < order.index("cache_stop")
        assert order.index("cache_stop") < order.index("db_stop")

    def test_status_health_consistency(self) -> None:
        """status and health should agree on subsystem state."""
        reg = SubsystemRegistry()
        reg.register("good", _StubSubsystem(name="good"))
        reg.register("bad", _StubSubsystem(name="bad", fail_on_start=1))
        reg.start_all()
        st = reg.status()
        h = reg.health()
        for name in st:
            is_healthy = st[name]["state"] == "started" and not st[name]["error"]
            assert h[name] == is_healthy

    def test_mixed_start_stop_methods(self) -> None:
        """Mix of subsystems with and without start/stop methods."""
        reg = SubsystemRegistry()
        with_start = _StubSubsystem(name="with_start")
        without = _StubNoMethods()
        reg.register("with_start", with_start)
        reg.register("without", without)
        assert reg.start_all() == 2
        assert with_start.start_count == 1
        assert reg.stop_all() == 2
        assert with_start.stop_count == 1
        assert reg._subsystems["without"].state == "stopped"

    def test_thread_safety(self) -> None:
        """Registering and starting from multiple threads should not corrupt state."""
        reg = SubsystemRegistry()
        n = 20
        results: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _register(idx: int) -> None:
            try:
                reg.register(f"t{idx}", _StubSubsystem(name=f"t{idx}"))
                with lock:
                    results.append(idx)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=_register, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(reg._subsystems) == n
        count = reg.start_all()
        assert count == n
        assert reg.all_healthy is True

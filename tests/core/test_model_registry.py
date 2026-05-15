"""Tests for ModelRegistry."""

import threading
import pytest

from distllm.core.model_registry import ModelEntry, ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_register_single_model(self):
        """Register one model, verify entry."""
        reg = ModelRegistry(max_models=4)
        entry = reg.register("model-a", "/path/a", 32)

        assert entry.name == "model-a"
        assert entry.path == "/path/a"
        assert entry.total_layers == 32
        assert entry.registered_at > 0

    def test_register_max_models(self):
        """Register up to max_models, verify rejection."""
        reg = ModelRegistry(max_models=2)
        reg.register("m1", "/p1", 10)
        reg.register("m2", "/p2", 20)

        with pytest.raises(ValueError, match="Maximum models"):
            reg.register("m3", "/p3", 30)

    def test_get_existing_model(self):
        """Get returns correct ModelEntry."""
        reg = ModelRegistry()
        reg.register("model-a", "/path/a", 32)

        entry = reg.get("model-a")
        assert entry is not None
        assert entry.name == "model-a"
        assert entry.path == "/path/a"

    def test_get_nonexistent_model(self):
        """Get returns None for unknown."""
        reg = ModelRegistry()
        assert reg.get("unknown") is None

    def test_list_models(self):
        """List returns all registered."""
        reg = ModelRegistry()
        reg.register("m1", "/p1", 10)
        reg.register("m2", "/p2", 20)

        models = reg.list_models()
        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"m1", "m2"}

    def test_default_model_setter(self):
        """Set and get default_model."""
        reg = ModelRegistry()
        reg.register("m1", "/p1", 10)
        reg.register("m2", "/p2", 20)

        reg.default_model = "m2"
        assert reg.default_model == "m2"

    def test_default_model_setter_invalid(self):
        """Setting default to unregistered model raises."""
        reg = ModelRegistry()
        reg.register("m1", "/p1", 10)

        with pytest.raises(ValueError, match="not registered"):
            reg.default_model = "unknown"

    def test_default_model_auto(self):
        """First registered becomes default if not set."""
        reg = ModelRegistry()
        reg.register("m1", "/p1", 10)
        assert reg.default_model == "m1"

    def test_is_registered(self):
        """True/False for known/unknown."""
        reg = ModelRegistry()
        reg.register("m1", "/p1", 10)

        assert reg.is_registered("m1") is True
        assert reg.is_registered("m2") is False

    def test_remove_model(self):
        """Remove and verify gone."""
        reg = ModelRegistry()
        reg.register("m1", "/p1", 10)
        reg.register("m2", "/p2", 20)

        assert reg.remove("m1") is True
        assert reg.is_registered("m1") is False
        assert reg.is_registered("m2") is True
        # Default should switch to remaining model
        assert reg.default_model == "m2"

    def test_remove_nonexistent(self):
        """Remove returns False for unknown."""
        reg = ModelRegistry()
        assert reg.remove("unknown") is False

    def test_thread_safety(self):
        """Concurrent register/get from multiple threads."""
        reg = ModelRegistry(max_models=100)
        errors = []

        def register_range(start, count):
            try:
                for i in range(count):
                    reg.register(f"m-{start}-{i}", f"/p-{start}-{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_range, args=(0, 25)),
            threading.Thread(target=register_range, args=(1, 25)),
            threading.Thread(target=register_range, args=(2, 25)),
            threading.Thread(target=register_range, args=(3, 25)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(reg.list_models()) == 100

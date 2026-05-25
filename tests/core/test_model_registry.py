"""Tests for ModelRegistry."""

import threading
import pytest

from distllm.core.model_registry import ModelEntry, ModelRegistry, ModelNotFoundError


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_register_single_model(self):
        reg = ModelRegistry(max_models=4)
        ver = reg.register_version("model-a", "1", "/path/a", 32)
        entry = reg.get("model-a")

        assert entry is not None
        assert entry.name == "model-a"
        assert "1" in entry.versions
        assert ver.path == "/path/a"
        assert ver.total_layers == 32
        assert ver.registered_at > 0

    def test_register_max_models(self):
        reg = ModelRegistry(max_models=2)
        reg.register_version("m1", "1", "/p1", 10)
        reg.register_version("m2", "1", "/p2", 20)

        with pytest.raises(ValueError, match="Maximum models"):
            reg.register_version("m3", "1", "/p3", 30)

    def test_get_existing_model(self):
        reg = ModelRegistry()
        reg.register_version("model-a", "1", "/path/a", 32)

        entry = reg.get("model-a")
        assert entry is not None
        assert entry.name == "model-a"

    def test_get_nonexistent_model(self):
        reg = ModelRegistry()
        assert reg.get("unknown") is None

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register_version("m1", "1", "/p1", 10)
        reg.register_version("m2", "1", "/p2", 20)

        names = reg.list_models()
        assert len(names) == 2
        assert set(names) == {"m1", "m2"}

    def test_default_model_setter(self):
        reg = ModelRegistry()
        reg.register_version("m1", "1", "/p1", 10)
        reg.register_version("m2", "1", "/p2", 20)

        reg.default_model = "m2"
        assert reg.default_model == "m2"

    def test_default_model_setter_invalid(self):
        reg = ModelRegistry()
        reg.register_version("m1", "1", "/p1", 10)

        with pytest.raises(ModelNotFoundError):
            reg.default_model = "unknown"

    def test_default_model_auto(self):
        reg = ModelRegistry()
        reg.register_version("m1", "1", "/p1", 10)
        assert reg.default_model == "m1"

    def test_is_registered(self):
        reg = ModelRegistry()
        reg.register_version("m1", "1", "/p1", 10)

        assert reg.is_registered("m1") is True
        assert reg.is_registered("m2") is False

    def test_remove_model(self):
        reg = ModelRegistry()
        reg.register_version("m1", "1", "/p1", 10)
        reg.register_version("m2", "1", "/p2", 20)

        assert reg.remove_model("m1") is True
        assert reg.is_registered("m1") is False
        assert reg.is_registered("m2") is True
        assert reg.default_model == "m2"

    def test_remove_nonexistent(self):
        reg = ModelRegistry()
        assert reg.remove_model("unknown") is False

    def test_thread_safety(self):
        reg = ModelRegistry(max_models=100)
        errors = []

        def register_range(start, count):
            try:
                for i in range(count):
                    reg.register_version(f"m-{start}-{i}", "1", f"/p-{start}-{i}", i)
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

"""Tests for VLLMNodeAdapter using real objects via load_module pattern.

NOTE: vLLM is not installed in the test env, so ``is_available()`` returns
False and ``load_model()`` raises ``ImportError``.  We test construction,
metadata methods, and error paths that exercise real code paths.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_vllm_mod = load_module("distllm/core/vllm_backend.py")
VLLMNodeAdapter = _vllm_mod.VLLMNodeAdapter

# Also load the real backends.vllm_backend for _validate_model_name
_backend_mod = load_module("distllm/backends/vllm_backend.py")
_validate_model_name = _backend_mod._validate_model_name


class TestValidateModelName:
    """_validate_model_name rejects dangerous model names."""

    def test_valid_org_name(self) -> None:
        # Should not raise
        _validate_model_name("meta-llama/Llama-2-7b")

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            _validate_model_name("../../etc/passwd")

    def test_rejects_absolute_path(self) -> None:
        with pytest.raises(ValueError, match="absolute path"):
            _validate_model_name("/etc/passwd")

    def test_rejects_bare_name_without_org(self) -> None:
        with pytest.raises(ValueError, match="org/name"):
            _validate_model_name("Llama-2-7b")

    def test_rejects_windows_path(self) -> None:
        with pytest.raises(ValueError, match="Windows path"):
            _validate_model_name("C:\\models\\llama.bin")

    def test_rejects_home_dir(self) -> None:
        with pytest.raises(ValueError, match="home-directory"):
            _validate_model_name("~/models/llama")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError):
            _validate_model_name("")


class TestVLLMNodeAdapterConstruction:
    """VLLMNodeAdapter construction and metadata."""

    def test_minimal_construction(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        assert adapter.model_name == "org/model"
        assert adapter.layer_start is None
        assert adapter.layer_end is None

    def test_with_layer_range(self) -> None:
        adapter = VLLMNodeAdapter(
            model_name="org/model", layer_start=0, layer_end=12,
        )
        assert adapter.layer_start == 0
        assert adapter.layer_end == 12

    def test_display_name(self) -> None:
        assert VLLMNodeAdapter.display_name() == "vLLM"

    def test_is_available(self) -> None:
        # vLLM not installed in test env
        assert VLLMNodeAdapter.is_available() is False

    def test_priority_for_cuda(self) -> None:
        assert VLLMNodeAdapter.priority_for("cuda") == 10

    def test_priority_for_cpu(self) -> None:
        # cpu is listed as 0 in VLLMNodeAdapter.priority_for
        # Let's check the actual value
        prio = VLLMNodeAdapter.priority_for("cpu")
        assert prio == 0

    def test_forward_before_load_raises(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        from distllm.backends.protocol import BackendState
        # The abstract forward raises RuntimeError when state != READY
        assert adapter.state == BackendState.UNINITIALIZED

    def test_load_model_raises_import_error(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        with pytest.raises((ImportError, ModuleNotFoundError)):
            adapter.load_model()

    def test_shutdown_idempotent(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        # shutdown when not loaded should not raise
        adapter.shutdown()

    def test_get_model_info_no_llm(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        info = adapter.get_model_info()
        assert info["backend"] == "vllm"
        assert info["model_name"] == "org/model"

    def test_get_metrics_no_llm(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        metrics = adapter.get_metrics()
        assert metrics["backend"] == "vllm"
        assert metrics["model_name"] == "org/model"

    def test_version_classmethod(self) -> None:
        version = VLLMNodeAdapter.version()
        assert isinstance(version, str)

    def test_description_classmethod(self) -> None:
        desc = VLLMNodeAdapter.description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_probe_health(self) -> None:
        # Class method, no instance needed
        assert VLLMNodeAdapter.probe_health() is True

    def test_health_check(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        assert adapter.health_check() is True

    def test_current_load(self) -> None:
        adapter = VLLMNodeAdapter(model_name="org/model")
        assert adapter.current_load() == 0.0

    def test_vllm_config_passed(self) -> None:
        adapter = VLLMNodeAdapter(
            model_name="org/model",
            vllm_config={"max_num_seqs": 8, "gpu_memory_utilization": 0.9},
        )
        assert adapter._config["max_num_seqs"] == 8
        assert adapter._config["gpu_memory_utilization"] == 0.9

    def test_trust_remote_code_in_config(self) -> None:
        adapter = VLLMNodeAdapter(
            model_name="org/model",
            trust_remote_code=True,
        )
        assert adapter._config.get("trust_remote_code") is True

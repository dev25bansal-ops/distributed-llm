"""Tests for LlamacppNodeAdapter using real objects via load_module pattern.

NOTE: llama-cpp-python is not installed in the test env, so ``is_available()``
returns False and ``load_model()`` raises ``ImportError``.  We test construction,
metadata methods, and error paths that exercise real code paths.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

# Load from the core re-export wrapper
_llamacpp_mod = load_module("distllm/core/llamacpp_backend.py")
LlamacppNodeAdapter = _llamacpp_mod.LlamacppNodeAdapter


class TestLlamacppNodeAdapterConstruction:
    """LlamacppNodeAdapter construction with various params."""

    def test_minimal_construction(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        assert adapter.model_path == "/models/model.gguf"
        assert adapter.n_gpu_layers == 0
        assert adapter.n_ctx == 2048
        assert adapter.seed == 0

    def test_custom_parameters(self) -> None:
        adapter = LlamacppNodeAdapter(
            model_path="/models/model.gguf",
            n_gpu_layers=20,
            n_ctx=4096,
            n_threads=8,
            n_batch=256,
            seed=42,
            verbose=True,
        )
        assert adapter.n_gpu_layers == 20
        assert adapter.n_ctx == 4096
        assert adapter.n_threads == 8
        assert adapter.n_batch == 256
        assert adapter.seed == 42
        assert adapter.verbose is True

    def test_with_layer_range(self) -> None:
        adapter = LlamacppNodeAdapter(
            model_path="/models/model.gguf",
            layer_start=0,
            layer_end=12,
        )
        assert adapter.layer_start == 0
        assert adapter.layer_end == 12

    def test_display_name(self) -> None:
        assert LlamacppNodeAdapter.display_name() == "llama.cpp"

    def test_is_available(self) -> None:
        assert LlamacppNodeAdapter.is_available() is False

    def test_priority_for_cpu(self) -> None:
        assert LlamacppNodeAdapter.priority_for("cpu") == 8

    def test_priority_for_mps(self) -> None:
        assert LlamacppNodeAdapter.priority_for("mps") == 9

    def test_priority_for_rocm(self) -> None:
        assert LlamacppNodeAdapter.priority_for("rocm") == 6

    def test_priority_for_cuda(self) -> None:
        assert LlamacppNodeAdapter.priority_for("cuda") == 4

    def test_priority_for_unknown(self) -> None:
        assert LlamacppNodeAdapter.priority_for("unknown") == 5

    def test_state_initial(self) -> None:
        from distllm.backends.protocol import BackendState
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        assert adapter.state == BackendState.UNINITIALIZED

    def test_load_model_raises_import_error(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        with pytest.raises((ImportError, ModuleNotFoundError)):
            adapter.load_model()

    def test_forward_before_load_raises(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        import torch
        with pytest.raises(RuntimeError, match="Cannot forward"):
            adapter.forward(input_ids=torch.zeros(1, 5, dtype=torch.long))

    def test_shutdown_idempotent(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        adapter.shutdown()


class TestLlamacppNodeAdapterExtraKwargs:
    """Extra kwargs forwarded to llama_cpp.Llama()."""

    def test_extra_kwargs_stored(self) -> None:
        adapter = LlamacppNodeAdapter(
            model_path="/models/model.gguf",
            use_mmap=False,
            use_mlock=True,
            offload_kqv=True,
        )
        assert adapter._extra_kwargs.get("use_mmap") is False
        assert adapter._extra_kwargs.get("use_mlock") is True
        assert adapter._extra_kwargs.get("offload_kqv") is True

    def test_extra_kwargs_empty_by_default(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        assert adapter._extra_kwargs == {}


class TestLlamacppNodeAdapterGenerateNotLoaded:
    """generate raises error before load_model."""

    def test_generate_raises(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        with pytest.raises(RuntimeError):
            adapter.generate("hello")

    def test_async_generate_raises(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        import asyncio
        with pytest.raises(RuntimeError):
            asyncio.run(adapter.async_generate("hello"))

    def test_get_tokenizer_before_load(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        assert adapter.get_tokenizer() is None


class TestLlamacppNodeAdapterVersion:
    """Class-level metadata methods."""

    def test_version(self) -> None:
        version = LlamacppNodeAdapter.version()
        assert isinstance(version, str)

    def test_probe_health(self) -> None:
        assert LlamacppNodeAdapter.probe_health() is True

    def test_health_check(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        assert adapter.health_check() is True

    def test_current_load(self) -> None:
        adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
        assert adapter.current_load() == 0.0

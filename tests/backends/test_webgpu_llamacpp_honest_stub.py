"""Tests for CQ-012–CQ-014 honest-stub surface.

Covers:
  - WebGPUNodeAdapter.generate() must raise NotImplementedError (no fabricated
    generation string) — backends/webgpu_backend.py:206
  - WebGPUNodeAdapter._forward_via_workers() must raise NotImplementedError
    (no zero logits) — backends/webgpu_backend.py:188
  - LlamacppNodeAdapter._forward_hidden_states() must raise NotImplementedError
    (no torch.zeros logits) when llama-cpp eval_logits is unavailable —
    backends/llamacpp_backend.py:148
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from distllm.backends.llamacpp_backend import LlamacppNodeAdapter
from distllm.backends.protocol import BackendState
from distllm.backends.webgpu_backend import WebGPUNodeAdapter
from distllm.errors import ModelLoadError


def _loaded_llamacpp_adapter(mock_llm: MagicMock) -> LlamacppNodeAdapter:
    adapter = LlamacppNodeAdapter(model_path="/models/model.gguf")
    adapter._llm = mock_llm
    adapter.state = BackendState.READY
    return adapter


class TestWebGPUSubprocessRaising:
    """webgpu_backend.py generate/_forward_via_workers must fail loudly."""

    def test_forward_not_loaded_raises_model_load_error(self) -> None:
        adapter = WebGPUNodeAdapter(model_name="org/model")
        assert adapter._model_loaded is False
        with pytest.raises(ModelLoadError, match="not loaded"):
            adapter.forward(input_ids=torch.zeros(1, 3, dtype=torch.long))

    def test_forward_no_workers_raises_model_load_error(self) -> None:
        adapter = WebGPUNodeAdapter(model_name="org/model")
        adapter._model_loaded = True
        adapter._ready_workers = []
        with pytest.raises(ModelLoadError, match="No browser workers"):
            adapter.forward(input_ids=torch.zeros(1, 3, dtype=torch.long))

    def test_forward_via_workers_raises_not_implemented(self) -> None:
        adapter = WebGPUNodeAdapter(model_name="org/model")
        adapter._model_loaded = True
        adapter._ready_workers = ["w1"]
        with pytest.raises(NotImplementedError, match="placeholder"):
            adapter.forward(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def test_generate_raises_not_implemented_never_fabricated(self) -> None:
        adapter = WebGPUNodeAdapter(model_name="org/model")
        adapter._model_loaded = True
        adapter._ready_workers = ["w1"]
        with pytest.raises(NotImplementedError, match="placeholder"):
            adapter.generate("hello world")

    def test_generate_raises_not_implemented_without_workers(self) -> None:
        adapter = WebGPUNodeAdapter(model_name="org/model")
        with pytest.raises(NotImplementedError, match="placeholder"):
            adapter.generate("hello")

    def test_generate_no_bracket_stub_string(self) -> None:
        """Regression: old stub returned '[WebGPU generation via w1]'."""
        adapter = WebGPUNodeAdapter(model_name="org/model")
        adapter._model_loaded = True
        adapter._ready_workers = ["w1"]
        result = None
        raised = False
        try:
            result = adapter.generate("hello")
        except NotImplementedError:
            raised = True
        assert raised, "generate() must raise NotImplementedError, not fabricate output"
        assert result is None or "[WebGPU generation" not in result


class TestLlamacppZeroLogitsRemoved:
    """llamacpp_backend.py _forward_hidden_states must fail loudly on no logits."""

    def test_forward_hidden_states_no_eval_logits_raises(self) -> None:
        mock_llm = MagicMock()
        mock_llm.eval = MagicMock()
        # .eval_logits is not set -> AttributeError on access in real ctx;
        # simulate via a plain object without _ctx.eval_logits
        class _Ctx:
            pass

        mock_llm._ctx = _Ctx()
        adapter = _loaded_llamacpp_adapter(mock_llm)

        hidden = torch.randn(1, 4, 16)
        with pytest.raises(NotImplementedError, match="eval_logits"):
            adapter._forward_hidden_states(hidden)

    def test_forward_hidden_states_malformed_logits_raises(self) -> None:
        mock_llm = MagicMock()
        mock_llm.eval = MagicMock()
        mock_llm._ctx.eval_logits = [1.0, 2.0]  # too few for (1,4,vocab)
        adapter = _loaded_llamacpp_adapter(mock_llm)

        hidden = torch.randn(1, 4, 16)
        with pytest.raises(NotImplementedError, match="eval_logits"):
            adapter._forward_hidden_states(hidden)

    def test_forward_hidden_states_real_logits_pass_through(self) -> None:
        mock_llm = MagicMock()
        mock_llm.eval = MagicMock()
        # 1 token x 8 vocab logits
        mock_llm._ctx.eval_logits = [float(i) for i in range(8)]
        adapter = _loaded_llamacpp_adapter(mock_llm)

        hidden = torch.randn(1, 1, 16)  # n_tokens = 1
        logits, kv = adapter._forward_hidden_states(hidden)
        assert logits.shape == (1, 1, 8)
        assert logits.dtype == torch.float32
        assert kv == []
        # Not all zeros — must be the real eval_logits values
        assert not torch.allclose(logits, torch.zeros_like(logits))

    def test_source_has_no_torch_zeros_logits(self) -> None:
        """Static guard: the fabricated fallback must stay deleted."""
        import pathlib

        src = pathlib.Path(
            "src/distllm/backends/llamacpp_backend.py"
        ).read_text(encoding="utf-8")
        assert "torch.zeros(1, n_tokens" not in src

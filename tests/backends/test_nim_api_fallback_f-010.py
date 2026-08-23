"""Regression test for audit finding F-010.

NimNodeAdapter._forward_via_api used to fabricate a synthetic "logits"
tensor when no local_model was set: np.log(prob) scattered at
idx = hash(token_str) % size (Python str hash is salted per-process via
PYTHONHASHSEED, so identical tokens landed at different vocab indices
across processes), or a one-hot in a hardcoded 32000-wide tensor.
Downstream argmax/sampling silently consumed garbage instead of the
path failing loudly.

Fix: the no-local-model input_ids path now raises NotImplementedError
(same policy as WebGPUNodeAdapter / TGI adapter) because NIM's HTTP API
only exposes top_logprobs for the generated token, which cannot
reconstruct a real logit distribution.
"""

from unittest.mock import MagicMock

import pytest
import torch

from distllm.backends.nim_backend import NimNodeAdapter


def _make_adapter(**kwargs) -> NimNodeAdapter:
    """Build an adapter without calling load_model() (no network)."""
    defaults = dict(
        model_name="meta/llama3-8b-instruct",
        api_url="http://localhost:8000/v1",
    )
    defaults.update(kwargs)
    return NimNodeAdapter(**defaults)


class TestNimForwardNoLocalModelFailsLoudly:
    """input_ids forward without local_model must refuse, not fabricate."""

    def test_forward_input_ids_raises_not_implemented(self):
        adapter = _make_adapter()  # no local_model
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with pytest.raises(NotImplementedError, match="local model"):
            adapter.forward(input_ids=input_ids)

    def test_no_http_request_made(self):
        """The refusal must happen before any NIM API call."""
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._request = MagicMock(
            side_effect=AssertionError("HTTP request must not be issued")
        )
        input_ids = torch.tensor([[7, 8, 9]])
        with pytest.raises(NotImplementedError):
            adapter.forward(input_ids=input_ids)
        adapter._request.assert_not_called()

    def test_result_is_never_a_fabricated_tensor(self):
        """Even if a caller swallows exceptions, no tensor may be returned."""
        adapter = _make_adapter()
        result = None
        try:
            result = adapter.forward(input_ids=torch.tensor([[42]]))
        except NotImplementedError:
            pass
        assert result is None

    def test_error_message_mentions_top_logprobs_limitation(self):
        adapter = _make_adapter()
        with pytest.raises(NotImplementedError, match="top_logprobs"):
            adapter.forward(input_ids=torch.tensor([[1]]))

    def test_hash_scatter_path_is_gone(self):
        """The hash()-based fabrication code must not exist anymore."""
        import inspect

        src = inspect.getsource(NimNodeAdapter)
        assert "hash(token_str)" not in src


class TestNimForwardLocalModelStillWorks:
    """The local-model path must be unaffected by the fix."""

    def test_local_model_forward_returns_logits(self):
        model = MagicMock()
        out = MagicMock(spec=["logits", "past_key_values"])
        out.logits = torch.randn(1, 4, 32)
        out.past_key_values = []
        model.return_value = out

        adapter = _make_adapter(local_model=model)
        logits, kv = adapter.forward(input_ids=torch.tensor([[1, 2, 3, 4]]))
        assert logits.shape == (1, 4, 32)
        assert kv == []

    def test_hidden_states_without_local_model_still_refuses(self):
        """Pipeline-mode hidden_states forward already refused pre-fix."""
        adapter = _make_adapter()
        with pytest.raises(NotImplementedError):
            adapter.forward(hidden_states=torch.randn(1, 4, 8))


class TestGenerationUnaffected:
    """generate() (text generation over HTTP) must remain available."""

    def test_generate_requires_session_only(self):
        adapter = _make_adapter()
        with pytest.raises(Exception, match="not connected|load_model"):
            adapter.generate("hi")

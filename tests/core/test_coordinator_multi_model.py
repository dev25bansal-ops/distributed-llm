"""Tests for coordinator multi-model integration."""

import pytest
from unittest.mock import MagicMock, patch

from distllm.core.coordinator import Coordinator


class TestCoordinatorMultiModel:
    """Tests for multi-model support in Coordinator."""

    def _make_minimal_coordinator(self):
        """Create a coordinator with minimal mocking (no GPU needed)."""
        with patch("distllm.core.coordinator.AutoTokenizer") as mock_tokenizer_cls:

            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
            mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: "hello"
            mock_tokenizer.eos_token_id = 0
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            coord = Coordinator(
                model_name="test-model",
                port=50050,
                dtype="float16",
            )
            coord.tokenizer = mock_tokenizer
            return coord

    def test_no_multi_model_returns_single(self):
        coord = self._make_minimal_coordinator()
        assert coord._model_registry is None
        assert coord.list_models() == ["test-model"]

    def test_get_model_name_fallback(self):
        coord = self._make_minimal_coordinator()
        assert coord.get_model_name() == "test-model"
        assert coord.get_model_name("other") == "test-model"

    def test_register_model_creates_registry(self):
        coord = self._make_minimal_coordinator()
        coord.register_model("model-b", "/path/b", 24)
        assert coord._model_registry is not None
        assert "model-b" in coord.list_models()

    def test_model_name_resolution_with_registry(self):
        coord = self._make_minimal_coordinator()
        coord.register_model("model-a", "/path/a", 32)
        coord.register_model("model-b", "/path/b", 24)

        assert coord.get_model_name("model-b") == "model-b"
        assert coord.get_model_name() == "model-a"
        assert coord.get_model_name("unknown") == "model-a"

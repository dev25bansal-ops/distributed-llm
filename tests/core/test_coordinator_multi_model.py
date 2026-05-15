"""Tests for coordinator multi-model integration."""

import pytest
from unittest.mock import MagicMock, patch

from distllm.core.coordinator import Coordinator


class TestCoordinatorMultiModel:
    """Tests for multi-model support in Coordinator."""

    def _make_minimal_coordinator(self):
        """Create a coordinator with minimal mocking (no GPU needed)."""
        with patch("distllm.core.coordinator.AutoTokenizer") as mock_tokenizer_cls, \
             patch("distllm.core.coordinator.GRPCServer"), \
             patch("distllm.core.coordinator.ModelPartitioner") as mock_partitioner_cls:

            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
            mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: "hello"
            mock_tokenizer.eos_token_id = 0
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            mock_partitioner = MagicMock()
            mock_partitioner_cls.return_value = mock_partitioner

            coord = Coordinator(
                model_name="test-model",
                port=50050,
                dtype="float16",
            )
            coord.tokenizer = mock_tokenizer
            return coord

    def test_no_multi_model_returns_single(self):
        """Without multi-model config, list_models returns [model_name]."""
        coord = self._make_minimal_coordinator()

        assert coord._model_registry is None
        assert coord.list_models() == ["test-model"]

    def test_get_model_name_fallback(self):
        """Without registry, get_model_name returns self.model_name."""
        coord = self._make_minimal_coordinator()

        assert coord.get_model_name() == "test-model"
        assert coord.get_model_name("other") == "test-model"

    def test_register_model_creates_registry(self):
        """register_model() creates registry if None."""
        coord = self._make_minimal_coordinator()

        coord.register_model("model-b", "/path/b", 24)

        assert coord._model_registry is not None
        assert "model-b" in coord.list_models()

    def test_model_name_resolution_with_registry(self):
        """With registry, resolves requested > default > fallback."""
        coord = self._make_minimal_coordinator()
        coord.register_model("model-a", "/path/a", 32)
        coord.register_model("model-b", "/path/b", 24)

        # Requested model exists
        assert coord.get_model_name("model-b") == "model-b"
        # Default fallback
        assert coord.get_model_name() == "model-a"  # first registered
        # Unknown falls back to default
        assert coord.get_model_name("unknown") == "model-a"

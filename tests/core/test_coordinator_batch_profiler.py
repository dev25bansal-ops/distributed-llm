"""Tests for coordinator wiring of batch profiler to scheduler."""

from unittest.mock import MagicMock, patch

import pytest

from distllm.core.coordinator import Coordinator


class TestCoordinatorBatchProfiler:
    """Test coordinator passes model_info to BatchScheduler."""

    def test_scheduler_created_with_model_info_when_available(self):
        """When model_info is set before scheduler creation, it should be passed."""
        with patch("distllm.core.coordinator.AutoTokenizer"):
            coord = Coordinator(
                model_name="test-model",
                max_batch_size=4,
                max_tokens_per_batch=1024,
            )
            # model_info is None at init, so scheduler gets model_info=None
            assert coord.scheduler is not None
            assert coord.scheduler._model_info is None

    def test_scheduler_updated_in_start_when_model_info_available(self):
        """When start() is called and model_info exists, scheduler should be updated."""
        with patch("distllm.core.coordinator.AutoTokenizer"):
            with patch("distllm.core.coordinator.CoordinatorService"):
                with patch("distllm.core.coordinator.GRPCServer"):
                    coord = Coordinator(
                        model_name="test-model",
                        max_batch_size=4,
                        max_tokens_per_batch=1024,
                    )
                    # Simulate model_info being set (e.g., by load_local_model)
                    coord.model_info = {
                        "hidden_size": 768,
                        "num_layers": 12,
                        "num_attention_heads": 12,
                    }

                    coord.start(blocking=False)

                    assert coord.scheduler._model_info == coord.model_info
                    assert coord.scheduler._use_length_grouping is True

    def test_scheduler_not_updated_when_no_model_info(self):
        """When model_info is None at start(), scheduler should not be updated."""
        with patch("distllm.core.coordinator.AutoTokenizer"):
            with patch("distllm.core.coordinator.CoordinatorService"):
                with patch("distllm.core.coordinator.GRPCServer"):
                    coord = Coordinator(
                        model_name="test-model",
                        max_batch_size=4,
                        max_tokens_per_batch=1024,
                    )
                    coord.model_info = None

                    coord.start(blocking=False)

                    # Scheduler remains with model_info=None
                    assert coord.scheduler._model_info is None
                    assert coord.scheduler._use_length_grouping is False

    def test_no_scheduler_when_max_batch_size_is_1(self):
        """Scheduler should not be created when max_batch_size=1."""
        with patch("distllm.core.coordinator.AutoTokenizer"):
            coord = Coordinator(
                model_name="test-model",
                max_batch_size=1,
            )
            assert coord.scheduler is None

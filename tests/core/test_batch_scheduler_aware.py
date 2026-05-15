"""Tests for model-aware batch scheduling."""

import pytest
import torch
from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus


class TestBatchSchedulerModelAware:
    """Test BatchScheduler with model_info for model-aware scheduling."""

    def test_init_with_model_info(self):
        model_info = {
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
        }
        scheduler = BatchScheduler(
            max_batch_size=4,
            max_tokens_per_batch=1024,
            model_info=model_info,
        )
        assert scheduler._model_info == model_info
        assert scheduler._use_length_grouping is True

    def test_init_without_model_info(self):
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1024)
        assert scheduler._model_info is None
        assert scheduler._use_length_grouping is False

    def test_schedule_includes_batch_tags(self):
        model_info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        scheduler = BatchScheduler(
            max_batch_size=4,
            max_tokens_per_batch=1024,
            model_info=model_info,
        )

        seq = Sequence(request_id="req-1", prompt_tokens=list(range(10)))
        scheduler.add(seq)

        batch = scheduler.schedule()
        assert batch is not None
        assert "avg_seq_len" in batch.batch_tags
        assert "avg_tokens_remaining" in batch.batch_tags
        assert batch.batch_tags["avg_tokens_remaining"] == 256  # default max_new_tokens - 0 generated

    def test_batch_tags_without_model_info(self):
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1024)
        seq = Sequence(request_id="req-1", prompt_tokens=list(range(10)))
        scheduler.add(seq)

        batch = scheduler.schedule()
        assert batch is not None
        # avg_tokens_remaining is always tracked
        assert "avg_tokens_remaining" in batch.batch_tags
        # avg_seq_len only present when model_info is set
        assert "avg_seq_len" not in batch.batch_tags

    def test_batch_tags_update_after_step(self):
        scheduler = BatchScheduler(
            max_batch_size=4,
            max_tokens_per_batch=1024,
            model_info={"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12},
        )

        seq = Sequence(request_id="req-1", prompt_tokens=list(range(10)), max_new_tokens=5)
        scheduler.add(seq)

        batch = scheduler.schedule()
        next_tokens = torch.tensor([42])
        scheduler.step(batch, next_tokens)

        # Schedule again
        batch2 = scheduler.schedule()
        assert batch2 is not None
        # avg_tokens_remaining should decrease
        assert batch2.batch_tags["avg_tokens_remaining"] < batch.batch_tags["avg_tokens_remaining"]

    def test_length_based_grouping_variances(self):
        model_info = {"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}
        scheduler = BatchScheduler(
            max_batch_size=8,
            max_tokens_per_batch=4096,
            model_info=model_info,
        )

        # Add sequences of varying lengths
        scheduler.add(Sequence(request_id="short", prompt_tokens=list(range(5))))
        scheduler.add(Sequence(request_id="medium", prompt_tokens=list(range(50))))
        scheduler.add(Sequence(request_id="long", prompt_tokens=list(range(200))))

        batch = scheduler.schedule()
        assert batch is not None
        assert batch.batch_tags["length_variance"] > 0

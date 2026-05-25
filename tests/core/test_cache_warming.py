"""Tests for CacheWarmer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.cache_warming import CacheWarmer


class TestCacheWarming:
    """Tests for CacheWarmer."""

    def test_warm_prompts(self):
        """Warms N prompts."""
        warmer = CacheWarmer()
        coord = MagicMock()
        coord.generate.return_value = "test"

        count = warmer.warm(["prompt1", "prompt2", "prompt3"], coord)

        assert count == 3
        assert coord.generate.call_count == 3

    def test_warm_from_file_list(self):
        """Loads JSON list of prompts."""
        warmer = CacheWarmer()
        coord = MagicMock()
        coord.generate.return_value = "test"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["p1", "p2"], f)
            f.flush()
            count = warmer.warm_from_file(f.name, coord)

        assert count == 2
        Path(f.name).unlink()

    def test_warm_from_file_dict(self):
        """Loads JSON with 'prompts' key."""
        warmer = CacheWarmer()
        coord = MagicMock()
        coord.generate.return_value = "test"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"prompts": ["p1", "p2", "p3"]}, f)
            f.flush()
            count = warmer.warm_from_file(f.name, coord)

        assert count == 3
        Path(f.name).unlink()

    def test_warm_from_file_missing(self):
        """Raises FileNotFoundError."""
        warmer = CacheWarmer()
        coord = MagicMock()

        with pytest.raises(FileNotFoundError):
            warmer.warm_from_file("/nonexistent/path.json", coord)

    def test_warm_handles_errors(self):
        """Failed prompts don't stop warming."""
        warmer = CacheWarmer()
        coord = MagicMock()
        coord.generate.side_effect = [
            "ok",
            Exception("fail"),
            "ok",
        ]

        count = warmer.warm(["p1", "p2", "p3"], coord)

        assert count == 2  # p1 and p3 succeeded


class TestCacheWarmingTiers:
    def test_add_tier_creates_tier(self):
        warmer = CacheWarmer()
        warmer.add_tier("hot", ["p1", "p2"], capture_cuda_graphs=True)
        assert len(warmer._tiers) == 1
        assert warmer._tiers[0].name == "hot"
        assert warmer._tiers[0].prompts == ["p1", "p2"]
        assert warmer._tiers[0].capture_cuda_graphs is True

    def test_add_tier_default_batch_sizes(self):
        warmer = CacheWarmer()
        warmer.add_tier("cold", ["p1"])
        assert warmer._tiers[0].batch_sizes == [1, 2, 4, 8, 16, 32]

    def test_run_single_tier(self):
        warmer = CacheWarmer()
        warmer.add_tier("hot", ["p1", "p2"])
        coord = MagicMock()
        coord.generate.return_value = "ok"
        coord.scheduler = None
        stats = warmer.run(coord)
        assert stats.warmed == 2
        assert stats.failed == 0
        assert coord.generate.call_count == 2

    def test_run_multiple_tiers(self):
        warmer = CacheWarmer()
        warmer.add_tier("hot", ["p1", "p2"])
        warmer.add_tier("warm", ["p3"])
        coord = MagicMock()
        coord.generate.return_value = "ok"
        coord.scheduler = None
        stats = warmer.run(coord)
        assert stats.warmed == 3
        assert stats.total_prompts == 3

    def test_run_handles_partial_failures(self):
        warmer = CacheWarmer()
        warmer.add_tier("hot", ["p1", "p2", "p3"])
        coord = MagicMock()
        coord.generate.side_effect = ["ok", Exception("fail"), "ok"]
        coord.scheduler = None
        stats = warmer.run(coord)
        assert stats.warmed == 2
        assert stats.failed == 1

    def test_run_returns_stats(self):
        warmer = CacheWarmer()
        warmer.add_tier("hot", ["p1"])
        coord = MagicMock()
        coord.generate.return_value = "ok"
        coord.scheduler = None
        stats = warmer.run(coord)
        assert isinstance(stats.warmed, int)
        assert isinstance(stats.duration_seconds, float)
        assert stats.duration_seconds >= 0

    def test_get_stats_after_run(self):
        warmer = CacheWarmer()
        warmer.add_tier("hot", ["p1"])
        coord = MagicMock()
        coord.generate.return_value = "ok"
        coord.scheduler = None
        warmer.run(coord)
        stats = warmer.get_stats()
        assert stats.warmed == 1

    def test_get_stats_before_run(self):
        warmer = CacheWarmer()
        stats = warmer.get_stats()
        assert stats.warmed == 0

    def test_from_config_creates_tiers(self):
        config = {
            "tiers": [
                {"name": "hot", "prompts": ["p1", "p2"], "capture_cuda_graphs": True},
                {"name": "warm", "prompts": ["p3"]},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        warmers = CacheWarmer.from_config(f.name)
        Path(f.name).unlink(missing_ok=True)
        assert len(warmers._tiers) == 2
        assert warmers._tiers[0].name == "hot"
        assert warmers._tiers[1].name == "warm"

    def test_from_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CacheWarmer.from_config("/nonexistent/config.json")

    def test_from_config_with_prompts_file(self):
        warmer = CacheWarmer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
            json.dump(["x", "y", "z"], pf)
        config = {"tiers": [{"name": "cold", "prompts_file": pf.name}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf:
            json.dump(config, cf)
        warmer = CacheWarmer.from_config(cf.name)
        Path(cf.name).unlink(missing_ok=True)
        Path(pf.name).unlink(missing_ok=True)
        assert len(warmer._tiers) == 1
        assert warmer._tiers[0].prompts == ["x", "y", "z"]


class TestCacheTierExecution:
    """Hot/warm tier execution order and properties."""

    def test_hot_tier_has_cuda_graphs(self):
        warmer = CacheWarmer()
        warmer.add_tier("hot", ["p1", "p2"], capture_cuda_graphs=True)
        assert warmer._tiers[0].capture_cuda_graphs is True

    def test_warm_tier_no_cuda_graphs_by_default(self):
        warmer = CacheWarmer()
        warmer.add_tier("warm", ["p3", "p4"])
        assert warmer._tiers[0].capture_cuda_graphs is False

    def test_hot_executed_before_warm(self):
        warmer = CacheWarmer()
        execution_order = []

        coord = MagicMock()
        def track(prompt, **kw):
            execution_order.append(prompt)
            return "ok"
        coord.generate.side_effect = track
        coord.scheduler = None

        warmer.add_tier("hot", ["hot1", "hot2"])
        warmer.add_tier("warm", ["warm1"])
        warmer.run(coord)

        assert execution_order == ["hot1", "hot2", "warm1"]

    def test_multiple_tiers_all_warmed(self):
        warmer = CacheWarmer()
        coord = MagicMock()
        coord.generate.return_value = "ok"
        coord.scheduler = None

        warmer.add_tier("hot", ["a", "b", "c"])
        warmer.add_tier("warm", ["d", "e"])
        warmer.add_tier("cold", ["f"])
        stats = warmer.run(coord)

        assert stats.warmed == 6
        assert stats.total_prompts == 6

    def test_tier_failure_does_not_stop_subsequent(self):
        warmer = CacheWarmer()
        coord = MagicMock()
        coord.generate.side_effect = ["ok", Exception("fail"), "ok"]
        coord.scheduler = None

        warmer.add_tier("hot", ["p1"])
        warmer.add_tier("warm", ["p2", "p3"])
        stats = warmer.run(coord)

        assert stats.warmed == 2
        assert stats.failed == 1

    def test_cuda_graph_captured_only_for_hot(self):
        warmer = CacheWarmer()
        coord = MagicMock()
        coord.generate.return_value = "ok"
        coord.scheduler = None

        warmer.add_tier("hot", ["p1"], capture_cuda_graphs=True)
        warmer.add_tier("warm", ["p2"], capture_cuda_graphs=False)

        with patch("torch.cuda.is_available", return_value=True):
            with patch.object(warmer, "_capture_cuda_graphs_for_tier", return_value=1) as mock_capture:
                stats = warmer.run(coord)
                # Should only try to capture for hot tier (warm has capture_cuda_graphs=False)
                assert mock_capture.call_count == 1
                assert stats.cuda_graphs_captured == 1

    def test_from_config_hot_warm_tiers(self):
        config = {
            "tiers": [
                {"name": "hot", "prompts": ["frequent"], "capture_cuda_graphs": True},
                {"name": "warm", "prompts": ["less_frequent"]},
                {"name": "cold", "prompts": ["rare"]},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        warmer = CacheWarmer.from_config(f.name)
        Path(f.name).unlink(missing_ok=True)

        assert warmer._tiers[0].name == "hot"
        assert warmer._tiers[0].capture_cuda_graphs is True
        assert warmer._tiers[1].name == "warm"
        assert warmer._tiers[1].capture_cuda_graphs is False
        assert warmer._tiers[2].name == "cold"

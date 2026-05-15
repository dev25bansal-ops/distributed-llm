"""Tests for CacheWarmer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

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

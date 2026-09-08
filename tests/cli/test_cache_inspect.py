"""Tests for ``distllm cache inspect <token_hash>`` CLI command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from typer.testing import CliRunner

from distllm.cli.cache import (
    _count_blocks,
    _inspect_manager,
    _MANAGER_REGISTRY,
    print_cache_inspect,
    register_cache_manager,
    run_cache_inspect,
)
from distllm.cli.main import app


runner = CliRunner()


class TestRunCacheInspect:
    """``run_cache_inspect`` looks up token_hash in registered managers."""

    def setup_method(self):
        # Clear the registry between tests so they don't leak.
        _MANAGER_REGISTRY.clear()

    def teardown_method(self):
        _MANAGER_REGISTRY.clear()

    def test_returns_none_when_no_manager_registered(self):
        assert run_cache_inspect("any-hash") is None

    def test_validates_non_empty_token_hash(self):
        with __import__("pytest").raises(ValueError):
            run_cache_inspect("")
        with __import__("pytest").raises(ValueError):
            run_cache_inspect("   ")

    def test_finds_entry_in_manager(self):
        manager = MagicMock()
        manager.caches = {"req-1": MagicMock()}
        manager._metadata = {
            "req-1": {"last_accessed": 1700000000.0, "created_at": 1699999000.0},
        }
        manager._request_prefix = {"req-1": "abc123"}
        manager.model_id = "llama-3-70b"

        # Build a real cache object with .cache list of length 5
        cache = MagicMock()
        cache.cache = [MagicMock()] * 5
        manager.caches = {"req-1": cache}

        register_cache_manager(manager)
        result = run_cache_inspect("abc123")
        assert result is not None
        assert result["model_id"] == "llama-3-70b"
        assert result["sequence_id"] == "req-1"
        assert result["block_count"] == 5
        assert result["last_access_ts"] == 1700000000.0
        assert result["prefix_hash"] == "abc123"

    def test_returns_none_for_unmatched_hash(self):
        manager = MagicMock()
        manager.caches = {"req-1": MagicMock()}
        manager._metadata = {"req-1": {"last_accessed": 1700000000.0}}
        manager._request_prefix = {"req-1": "abc123"}
        manager.model_id = "x"

        register_cache_manager(manager)
        assert run_cache_inspect("xyz999") is None

    def test_case_insensitive_match(self):
        manager = MagicMock()
        manager.caches = {"req-1": MagicMock()}
        manager._metadata = {"req-1": {"last_accessed": 1700000000.0}}
        manager._request_prefix = {"req-1": "AbC123"}
        manager.model_id = "x"
        cache = MagicMock()
        cache.cache = []
        manager.caches = {"req-1": cache}

        register_cache_manager(manager)
        result = run_cache_inspect("abc123")
        assert result is not None
        assert result["sequence_id"] == "req-1"

    def test_handles_missing_metadata_gracefully(self):
        """A manager with no _metadata dict must not crash the lookup."""
        manager = MagicMock()
        manager.caches = {"req-1": MagicMock()}
        manager._metadata = {}
        manager._request_prefix = {"req-1": "abc123"}
        manager.model_id = "x"

        register_cache_manager(manager)
        result = run_cache_inspect("abc123")
        assert result is not None
        assert result["last_access_ts"] == 0.0

    def test_register_cache_manager_dedups(self):
        manager = MagicMock()
        manager._request_prefix = {}
        manager.caches = {}
        manager._metadata = {}
        register_cache_manager(manager)
        register_cache_manager(manager)
        assert _MANAGER_REGISTRY.count(manager) == 1

    def test_iterates_multiple_managers(self):
        m1 = MagicMock()
        m1.caches = {}
        m1._metadata = {}
        m1._request_prefix = {"req-a": "hash-a"}
        m1.model_id = "model-a"

        m2 = MagicMock()
        m2.caches = {"req-b": MagicMock()}
        m2._metadata = {"req-b": {"last_accessed": 12345.0}}
        m2._request_prefix = {"req-b": "hash-b"}
        m2.model_id = "model-b"
        cache = MagicMock()
        cache.cache = [None, None]
        m2.caches = {"req-b": cache}

        register_cache_manager(m1)
        register_cache_manager(m2)

        # First manager doesn't have it, second does.
        result = run_cache_inspect("hash-b")
        assert result is not None
        assert result["model_id"] == "model-b"


class TestCountBlocks:
    def test_none_cache(self):
        assert _count_blocks(None) == 0

    def test_dict_cache(self):
        cache = MagicMock()
        cache.cache = {"a": 1, "b": 2, "c": 3}
        assert _count_blocks(cache) == 3

    def test_list_cache(self):
        cache = MagicMock()
        cache.cache = [1, 2, 3, 4]
        assert _count_blocks(cache) == 4


class TestPrintCacheInspect:
    """The printer must not raise on any reasonable input."""

    def test_prints_not_found(self):
        from rich.console import Console

        console = Console(record=True, width=120)
        print_cache_inspect("missing-hash", None, console)
        output = console.export_text()
        assert "missing-hash" in output
        assert "not found" in output.lower()

    def test_prints_entry(self):
        from rich.console import Console

        entry = {
            "model_id": "llama-3-70b",
            "layer_index": 0,
            "sequence_id": "req-42",
            "block_count": 7,
            "last_access_ts": 1700000000.0,
            "created_at": 1699999000.0,
            "prefix_hash": "abc",
            "request_id": "req-42",
        }
        console = Console(record=True, width=120)
        print_cache_inspect("abc", entry, console)
        output = console.export_text()
        assert "llama-3-70b" in output
        assert "req-42" in output
        assert "block_count" in output


class TestCacheCliCommand:
    """End-to-end CLI invocation."""

    def test_cache_inspect_help(self):
        result = runner.invoke(app, ["cache", "inspect", "--help"])
        assert result.exit_code == 0
        assert "token_hash" in result.output or "TOKEN_HASH" in result.output

    def test_cache_inspect_not_found(self):
        # Without a coordinator or registered manager, every lookup returns None.
        result = runner.invoke(app, ["cache", "inspect", "no-such-hash"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()

    def test_cache_inspect_json_output(self):
        result = runner.invoke(app, ["cache", "inspect", "no-such-hash", "--json"])
        assert result.exit_code == 0
        out = result.output
        # Find the JSON object embedded in rich output.
        start = out.find("{")
        end = out.rfind("}")
        assert start != -1 and end != -1 and end > start
        data = json.loads(out[start:end + 1])
        assert data["token_hash"] == "no-such-hash"
        assert data["entry"] is None
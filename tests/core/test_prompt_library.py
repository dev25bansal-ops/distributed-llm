"""Tests for the version-controlled prompt template library."""

import threading
import time
import tempfile
import os
from dataclasses import fields
from datetime import datetime, timezone

import pytest

from distllm.core.prompt_library import PromptRepository, PromptVersion


@pytest.fixture
def repo():
    db_path = os.path.join(tempfile.gettempdir(), f"test_prompts_{time.time_ns()}.db")
    r = PromptRepository(db_path=db_path)
    yield r
    r.close()
    for _ in range(5):  # retry on Windows
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
            break
        except PermissionError:
            time.sleep(0.1)


class TestPromptVersion:
    def test_default_values(self):
        v = PromptVersion(name="test", template="hello", version=1)
        assert v.name == "test"
        assert v.template == "hello"
        assert v.version == 1
        assert isinstance(v.created_at, float)
        assert v.variables == []
        assert v.tags == []

    def test_with_variables(self):
        v = PromptVersion(name="t", template="{name}", version=1, variables=["name"])
        assert v.variables == ["name"]


class TestCreate:
    def test_returns_version_1(self, repo):
        v = repo.create("test", "hello world")
        assert v.version == 1
        assert v.name == "test"

    def test_increments_version(self, repo):
        repo.create("test", "v1")
        v2 = repo.create("test", "v2")
        assert v2.version == 2

    def test_separate_names(self, repo):
        a = repo.create("a", "aaa")
        b = repo.create("b", "bbb")
        assert a.version == 1
        assert b.version == 1


class TestGet:
    def test_get_latest(self, repo):
        repo.create("x", "v1")
        v2 = repo.create("x", "v2")
        got = repo.get("x")
        assert got.version == v2.version
        assert got.template == "v2"

    def test_get_specific_version(self, repo):
        repo.create("x", "v1")
        v2 = repo.create("x", "v2")
        got = repo.get("x", version=1)
        assert got.template == "v1"

    def test_get_nonexistent(self, repo):
        assert repo.get("nonexistent") is None


class TestList:
    def test_list_all(self, repo):
        repo.create("a", "aaa")
        repo.create("b", "bbb")
        names = [p.name for p in repo.list()]
        assert "a" in names
        assert "b" in names

    def test_list_filter_by_name(self, repo):
        repo.create("target", "x")
        repo.create("other", "y")
        results = repo.list(name="target")
        assert len(results) == 1
        assert results[0].name == "target"


class TestDiff:
    def test_diff_returns_changes(self, repo):
        repo.create("x", "hello")
        repo.create("x", "world")
        d = repo.diff("x", 1, 2)
        assert d is not None
        assert "template" in d


class TestDelete:
    def test_delete_returns_true(self, repo):
        repo.create("x", "x")
        assert repo.delete("x") is True

    def test_delete_removes(self, repo):
        repo.create("x", "x")
        repo.delete("x")
        assert repo.get("x") is None


class TestConcurrency:
    def test_concurrent_create(self, repo):
        errors = []
        def writer():
            try:
                for i in range(20):
                    repo.create(f"concurrent-{threading.get_ident()}", f"v{i}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Errors: {errors}"

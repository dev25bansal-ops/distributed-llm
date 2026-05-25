"""Internal file utility tests: _safe_filename, _get_storage_path, _resolve_storage_path."""

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from distllm.api.persistent_store import get_data_dir
from distllm.api.routes.files import _safe_filename, _get_storage_path, _resolve_storage_path


@contextmanager
def _override_env(key: str, value: str | None):
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class TestSafeFilename:
    def test_plain_filename_preserved(self):
        assert _safe_filename("train.jsonl") == "train.jsonl"

    def test_path_traversal_stripped(self):
        assert _safe_filename("../../etc/passwd") == "passwd"

    def test_windows_path_traversal_stripped(self):
        assert _safe_filename("..\\..\\etc\\passwd") == "passwd"

    def test_absolute_path_stripped(self):
        assert _safe_filename("/etc/passwd") == "passwd"

    def test_nested_relative_path_stripped(self):
        assert _safe_filename("foo/bar/../baz.txt") == "baz.txt"

    def test_empty_filename_uses_uuid(self):
        result = _safe_filename("")
        assert len(result) > 0
        assert result.startswith("upload_")

    def test_spaces_only_filename_uses_uuid(self):
        result = _safe_filename("   ")
        assert len(result) > 0
        assert result.startswith("upload_")


class TestStoragePath:
    def test_get_storage_path_format(self, tmp_path):
        base = tmp_path / "custom_files"
        with _override_env("DISTLLM_FILE_DIR", str(base)):
            path = _get_storage_path("file-abc123", "train.jsonl")
        assert path.parent == base
        assert path.name == "file-abc123_train.jsonl"

    def test_get_storage_path_strips_traversal_in_filename(self, tmp_path):
        base = tmp_path / "files"
        with _override_env("DISTLLM_FILE_DIR", str(base)):
            path = _get_storage_path("file-x", "../../etc/passwd")
        assert path.parent == base
        assert path.name.startswith("file-x_passwd")

    def test_get_storage_path_default_base(self):
        path = _get_storage_path("file-d", "doc.txt")
        assert path.parent == get_data_dir() / "files"

    def test_resolve_path_uses_storage_path_field(self):
        file_obj = {"storage_path": str(Path("C:\\custom\\path\\file.txt")), "filename": "unused.txt"}
        path = _resolve_storage_path("file-id", file_obj)
        assert str(path) == str(Path("C:\\custom\\path\\file.txt"))

    def test_resolve_path_falls_back_when_no_storage_path(self, tmp_path):
        base = tmp_path / "fallback_dir"
        with _override_env("DISTLLM_FILE_DIR", str(base)):
            path = _resolve_storage_path("file-fb", {"filename": "fallback.txt"})
        assert path.parent == base
        assert path.name == "file-fb_fallback.txt"

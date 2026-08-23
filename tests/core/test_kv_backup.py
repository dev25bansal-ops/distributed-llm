"""Tests for KVBackupManager -- KV cache snapshot backup and restore.

Covers:
- Construction with backup directory
- create_snapshot with empty KVCacheManager
- list_backups returns entries
- restore raises ValueError for unknown backup
- delete_backup removes entry
- _save_manifest and _load_manifests

No MagicMock -- real temp directory and file operations.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/kv_backup.py")
KVBackupManager = _mod.KVBackupManager
BackupManifest = _mod.BackupManifest


class _EmptyKVCacheManager:
    """Minimal KVCacheManager with no caches."""

    def __init__(self: Any) -> None:
        self.caches: dict[str, Any] = {}


class _StubKVCache:
    """Stub KV cache with get_all."""

    def __init__(self: Any, num_layers: int = 1, seq_len: int = 4) -> None:
        self.num_layers = num_layers
        self.sequence_length = seq_len

    def get_all(self: Any) -> list:
        return []


class _StubManagerWithCaches:
    """Manager with some caches."""

    def __init__(self: Any) -> None:
        self.caches = {
            "req-1": _StubKVCache(num_layers=2, seq_len=4),
            "req-2": _StubKVCache(num_layers=2, seq_len=8),
        }


class TestBackupManifest:
    """BackupManifest dataclass."""

    def test_minimal_construction(self) -> None:
        m = BackupManifest(backup_id="abc", created_at=100.0, num_entries=0, total_bytes=0)
        assert m.backup_id == "abc"
        assert m.num_entries == 0
        assert m.metadata == {}

    def test_with_metadata(self) -> None:
        m = BackupManifest(
            backup_id="xyz", created_at=200.0,
            num_entries=2, total_bytes=1024,
            metadata={"version": 1},
        )
        assert m.metadata["version"] == 1


class TestKVBackupManagerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            assert mgr._backup_dir == Path(tmpdir)
            assert mgr._manifests == {}
            assert Path(tmpdir).exists()

    def test_bad_path_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "subdir" / "backups"
            mgr = KVBackupManager(backup_dir=str(new_dir))
            assert new_dir.exists()


class TestKVBackupManagerSnapshot:
    """Snapshot creation."""

    def test_create_snapshot_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            mgr_impl = _EmptyKVCacheManager()
            backup_id = mgr.create_snapshot(mgr_impl)
            assert isinstance(backup_id, str)
            assert len(backup_id) > 0
            manifests = mgr.list_backups()
            assert len(manifests) == 1
            assert manifests[0]["backup_id"] == backup_id

    def test_create_snapshot_with_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            stub_mgr = _StubManagerWithCaches()
            backup_id = mgr.create_snapshot(stub_mgr, metadata={"reason": "test"})
            manifests = mgr.list_backups()
            assert len(manifests) == 1
            assert manifests[0]["num_entries"] == 2
            assert manifests[0]["metadata"]["reason"] == "test"


class TestKVBackupManagerRestore:
    """Restore operations."""

    def test_restore_unknown_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            with pytest.raises(ValueError, match="not found"):
                mgr.restore("nonexistent", _EmptyKVCacheManager())

    def test_restore_empty_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            backup_id = mgr.create_snapshot(_EmptyKVCacheManager())
            count = mgr.restore(backup_id, _EmptyKVCacheManager())
            assert count == 0


class TestKVBackupManagerDelete:
    """Delete operations."""

    def test_delete_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            backup_id = mgr.create_snapshot(_EmptyKVCacheManager())
            assert len(mgr.list_backups()) == 1
            deleted = mgr.delete_backup(backup_id)
            assert deleted is True
            assert len(mgr.list_backups()) == 0

    def test_delete_nonexistent_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            assert mgr.delete_backup("nonexistent") is False


class TestKVBackupManagerPersistence:
    """Manifest persistence."""

    def test_save_and_load_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KVBackupManager(backup_dir=tmpdir)
            manifest = BackupManifest(
                backup_id="test123", created_at=100.0,
                num_entries=3, total_bytes=999,
                metadata={"key": "val"},
            )
            mgr._save_manifest(manifest)
            # Reload should pick up saved manifest
            mgr2 = KVBackupManager(backup_dir=tmpdir)
            assert "test123" in mgr2._manifests
            assert mgr2._manifests["test123"].num_entries == 3

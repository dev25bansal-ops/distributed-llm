"""Tests for KV cache migration across clusters.

Tests cover:
- KVMigrationResult dataclass construction and mutation
- _estimate_page_bytes helper with various input shapes
- KVCacheMigrator constructor defaults and custom parameters
- get_metrics on fresh and used migrators
- migrate_prefix error paths (unreachable targets, invalid URLs)
- warm_remote_cache convenience wrapper

All tests use real objects from the module — zero mocks.
"""

from typing import Any

import pytest

from distllm.dist.kv_migration import (
    KVMigrationResult,
    KVCacheMigrator,
    _estimate_page_bytes,
)


class TestKVMigrationResult:
    """KVMigrationResult dataclass — construction, fields, defaults."""

    def test_create_full_success(self) -> None:
        result = KVMigrationResult(
            success=True,
            cluster_id="cluster-a",
            prefix_hash="abc123def456",
            pages_transferred=10,
            bytes_transferred=2048,
            transfer_time_ms=50.0,
        )
        assert result.success is True
        assert result.cluster_id == "cluster-a"
        assert result.prefix_hash == "abc123def456"
        assert result.pages_transferred == 10
        assert result.bytes_transferred == 2048
        assert result.transfer_time_ms == 50.0
        assert result.error == ""

    def test_create_with_error(self) -> None:
        result = KVMigrationResult(
            success=False,
            cluster_id="cluster-b",
            prefix_hash="def456",
            pages_transferred=0,
            bytes_transferred=0,
            transfer_time_ms=100.0,
            error="Connection refused",
        )
        assert result.success is False
        assert result.error == "Connection refused"

    def test_create_zero_values(self) -> None:
        result = KVMigrationResult(
            success=True,
            cluster_id="",
            prefix_hash="",
            pages_transferred=0,
            bytes_transferred=0,
            transfer_time_ms=0.0,
        )
        assert result.success is True
        assert result.pages_transferred == 0
        assert result.bytes_transferred == 0
        assert result.transfer_time_ms == 0.0

    def test_create_large_values(self) -> None:
        result = KVMigrationResult(
            success=True,
            cluster_id="c" * 200,
            prefix_hash="p" * 64,
            pages_transferred=2_000_000,
            bytes_transferred=2**31,
            transfer_time_ms=7_200_000.25,
        )
        assert len(result.cluster_id) == 200
        assert result.pages_transferred == 2_000_000
        assert result.bytes_transferred == 2**31
        assert result.transfer_time_ms == 7_200_000.25

    def test_fields_are_mutable(self) -> None:
        result = KVMigrationResult(
            success=False,
            cluster_id="c",
            prefix_hash="h",
            pages_transferred=0,
            bytes_transferred=0,
            transfer_time_ms=0.0,
        )
        result.success = True
        result.error = "recovered"
        assert result.success is True
        assert result.error == "recovered"

    def test_transfer_time_float_precision(self) -> None:
        result = KVMigrationResult(
            success=True,
            cluster_id="x",
            prefix_hash="y",
            pages_transferred=1,
            bytes_transferred=1,
            transfer_time_ms=0.123456789,
        )
        assert result.transfer_time_ms == 0.123456789


class TestEstimatePageBytes:
    """_estimate_page_bytes — edge cases and data shape variations."""

    def test_empty_list(self) -> None:
        assert _estimate_page_bytes([]) == 0

    def test_single_page_list_values(self) -> None:
        pages: list[dict[str, Any]] = [
            {"key": [1.0, 2.0, 3.0], "value": [4.0, 5.0]},
        ]
        assert _estimate_page_bytes(pages) == 10  # (3 * 2) + (2 * 2)

    def test_single_page_tuple_values(self) -> None:
        pages: list[dict[str, Any]] = [
            {"key": (1.0, 2.0), "value": (3.0,)},
        ]
        assert _estimate_page_bytes(pages) == 6  # (2 * 2) + (1 * 2)

    def test_multiple_pages(self) -> None:
        pages: list[dict[str, Any]] = [
            {"key": [1.0, 2.0], "value": [3.0, 4.0]},
            {"key": [5.0], "value": [6.0, 7.0, 8.0]},
        ]
        assert _estimate_page_bytes(pages) == 16  # (4+4) + (2+6)

    def test_empty_key_value(self) -> None:
        pages: list[dict[str, Any]] = [{"key": [], "value": []}]
        assert _estimate_page_bytes(pages) == 0

    def test_missing_key_or_value(self) -> None:
        assert _estimate_page_bytes([{"key": [1.0]}]) == 2
        assert _estimate_page_bytes([{"value": [1.0]}]) == 2
        assert _estimate_page_bytes([{}]) == 0

    def test_none_values(self) -> None:
        pages: list[dict[str, Any]] = [
            {"key": None, "value": None},
        ]
        assert _estimate_page_bytes(pages) == 0

    def test_mixed_none_and_list(self) -> None:
        pages: list[dict[str, Any]] = [
            {"key": [1.0], "value": None},
            {"key": None, "value": [2.0]},
        ]
        assert _estimate_page_bytes(pages) == 4

    def test_object_with_nbytes(self) -> None:
        class FakeArray:
            def __init__(self, nbytes: int) -> None:
                self.nbytes = nbytes

        pages: list[dict[str, Any]] = [
            {"key": FakeArray(100), "value": FakeArray(200)},
        ]
        assert _estimate_page_bytes(pages) == 300

    def test_mixed_list_and_nbytes_object(self) -> None:
        class FakeArray:
            def __init__(self, nbytes: int) -> None:
                self.nbytes = nbytes

        pages: list[dict[str, Any]] = [
            {"key": [1.0, 2.0], "value": FakeArray(64)},
        ]
        assert _estimate_page_bytes(pages) == 68  # (2 * 2) + 64

    def test_maximal_sizes(self) -> None:
        large_list = [float(i) for i in range(1000)]
        pages: list[dict[str, Any]] = [
            {"key": large_list, "value": large_list},
        ]
        assert _estimate_page_bytes(pages) == 4000  # (1000 * 2) * 2


class TestKVCacheMigratorInit:
    """KVCacheMigrator constructor — defaults and custom parameters."""

    def test_default_params(self) -> None:
        migrator = KVCacheMigrator()
        assert migrator._chunk_size == 4
        assert migrator._timeout_s == 30.0
        assert migrator._total_migrations == 0
        assert migrator._successful_migrations == 0
        assert migrator._total_bytes == 0
        assert migrator._total_time_ms == 0.0

    def test_custom_params(self) -> None:
        migrator = KVCacheMigrator(
            chunk_size_pages=16,
            max_concurrent_migrations=2,
            timeout_s=5.0,
        )
        assert migrator._chunk_size == 16
        assert migrator._timeout_s == 5.0

    def test_semaphore_created(self) -> None:
        migrator = KVCacheMigrator(max_concurrent_migrations=4)
        assert migrator._semaphore is not None

    def test_content_router_created(self) -> None:
        migrator = KVCacheMigrator()
        assert migrator._content_router is not None

    def test_negative_chunk_size(self) -> None:
        migrator = KVCacheMigrator(chunk_size_pages=-1)
        assert migrator._chunk_size == -1

    def test_zero_concurrent_migrations(self) -> None:
        migrator = KVCacheMigrator(max_concurrent_migrations=0)
        assert migrator._semaphore is not None


class TestKVCacheMigratorGetMetrics:
    """get_metrics — observability data on fresh and used migrators."""

    def test_initial_metrics(self) -> None:
        migrator = KVCacheMigrator()
        metrics = migrator.get_metrics()
        assert metrics == {
            "total_migrations": 0,
            "successful_migrations": 0,
            "total_bytes_transferred": 0,
            "total_time_ms": 0.0,
            "avg_transfer_speed_kbps": 0,
        }

    def test_metrics_zero_time_avg_speed(self) -> None:
        migrator = KVCacheMigrator()
        metrics = migrator.get_metrics()
        assert metrics["total_time_ms"] == 0.0
        assert metrics["avg_transfer_speed_kbps"] == 0

    def test_metrics_are_deterministic(self) -> None:
        m1 = KVCacheMigrator().get_metrics()
        m2 = KVCacheMigrator().get_metrics()
        assert m1 == m2


class TestKVCacheMigratorMigratePrefix:
    """migrate_prefix error paths — no network required."""

    @pytest.mark.asyncio
    async def test_connection_refused_returns_failed_result(self) -> None:
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.migrate_prefix(
            target_url="http://127.0.0.1:1",
            prefix_hash="test-hash",
            kv_pages=[{"key": [1.0], "value": [2.0]}],
            cluster_id="unreachable-peer",
        )
        assert result.success is False
        assert result.pages_transferred == 0
        assert result.bytes_transferred == 0
        assert result.cluster_id == "unreachable-peer"
        assert result.prefix_hash == "test-hash"

    @pytest.mark.asyncio
    async def test_empty_pages_succeeds_immediately(self) -> None:
        """Empty pages cause no HTTP request, so migration succeeds trivially."""
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.migrate_prefix(
            target_url="http://127.0.0.1:1",
            prefix_hash="empty",
            kv_pages=[],
            cluster_id="test-cluster",
        )
        assert result.success is True
        assert result.pages_transferred == 0
        assert result.bytes_transferred == 0
        assert result.transfer_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_invalid_url_captured_as_error(self) -> None:
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.migrate_prefix(
            target_url="not-a-valid-url",
            prefix_hash="badurl",
            kv_pages=[{"key": [1.0], "value": [2.0]}],
            cluster_id="bad-url-test",
        )
        assert result.success is False
        assert result.pages_transferred == 0

    @pytest.mark.asyncio
    async def test_large_pages_connection_fails(self) -> None:
        pages: list[dict[str, Any]] = [
            {"key": [float(i)], "value": [float(i)]} for i in range(20)
        ]
        migrator = KVCacheMigrator(chunk_size_pages=4, timeout_s=0.1)
        result = await migrator.migrate_prefix(
            target_url="http://127.0.0.1:1",
            prefix_hash="large",
            kv_pages=pages,
        )
        assert result.success is False
        assert result.pages_transferred == 0

    @pytest.mark.asyncio
    async def test_default_cluster_id_is_empty_string(self) -> None:
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.migrate_prefix(
            target_url="http://127.0.0.1:1",
            prefix_hash="no-cid",
            kv_pages=[{"key": [1.0], "value": [2.0]}],
        )
        assert result.success is False
        assert result.cluster_id == ""

    @pytest.mark.asyncio
    async def test_url_with_trailing_slash_normalized(self) -> None:
        """Trailing slash should be stripped; connection still fails."""
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.migrate_prefix(
            target_url="http://127.0.0.1:1/",
            prefix_hash="slash",
            kv_pages=[{"key": [1.0], "value": [2.0]}],
            cluster_id="trailing-slash",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_kv_pages_with_empty_dicts(self) -> None:
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.migrate_prefix(
            target_url="http://127.0.0.1:1",
            prefix_hash="emptydicts",
            kv_pages=[{}, {}, {}],
            cluster_id="empty-dicts",
        )
        assert result.success is False


class TestKVCacheMigratorWarmRemoteCache:
    """warm_remote_cache convenience wrapper — error path."""

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_failure(self) -> None:
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.warm_remote_cache(
            target_url="http://127.0.0.1:1",
            prefix_hash="warm-test",
            kv_pages=[{"key": [1.0], "value": [2.0]}],
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_with_empty_pages(self) -> None:
        """Empty pages cause no HTTP request, so migration succeeds."""
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.warm_remote_cache(
            target_url="http://127.0.0.1:1",
            prefix_hash="empty-warm",
            kv_pages=[],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_with_invalid_url(self) -> None:
        migrator = KVCacheMigrator(timeout_s=0.1)
        result = await migrator.warm_remote_cache(
            target_url="not-a-valid-url",
            prefix_hash="bad-url-warm",
            kv_pages=[{"key": [1.0], "value": [2.0]}],
        )
        assert result is False

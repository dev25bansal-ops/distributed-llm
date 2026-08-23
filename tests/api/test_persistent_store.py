"""Tests for PersistentStore -- SQLite-backed CRUD for batches, files, fine-tuning."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from distllm.api.persistent_store import PersistentStore

# Saved reference to the original sqlite3.connect for patching purposes.
# The alias sqlite3.connect is what PersistentStore calls internally.
_ORIG_CONNECT = sqlite3.connect


def _close_pool(store: PersistentStore) -> None:
    """Close all connections in the store's pool to avoid ResourceWarning."""
    for conn in list(store._pool):
        try:
            conn.close()
        except Exception:
            pass
    store._pool.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    """Return a fresh :memory: PersistentStore for each test.

    Best for single-threaded CRUD tests.  ``:memory:`` databases are
    per-connection, so each SQLite connection sees an independent database.
    This is fine for single-threaded tests but will NOT work for multi-
    threaded tests -- use ``file_store`` for those.
    """
    s = PersistentStore(db_path=":memory:")
    yield s
    _close_pool(s)


@pytest.fixture
def file_store(tmp_path, monkeypatch):
    """Return a PersistentStore backed by a temp file on disk.

    Cross-thread connection use is enabled (``check_same_thread=False``)
    so that connection-pool and thread-safety tests can share connections
    returned to the pool across threads.
    """
    db_path = str(tmp_path / "test.db")

    def _patched_connect(*args, **kwargs):
        kwargs.setdefault("check_same_thread", False)
        return _ORIG_CONNECT(*args, **kwargs)

    monkeypatch.setattr("distllm.api.persistent_store.sqlite3.connect", _patched_connect)
    s = PersistentStore(db_path=db_path)
    yield s

    _close_pool(s)


# ======================================================================
# Batch CRUD
# ======================================================================


class TestBatchCRUD:
    def test_save_and_get_batch(self, store: PersistentStore) -> None:
        store.save_batch("batch-1", {"id": "batch-1", "status": "pending"})
        result = store.get_batch("batch-1")
        assert result is not None
        assert result["id"] == "batch-1"
        assert result["status"] == "pending"

    def test_get_missing_batch(self, store: PersistentStore) -> None:
        assert store.get_batch("nonexistent") is None

    def test_list_batches_empty(self, store: PersistentStore) -> None:
        assert store.list_batches() == []

    def test_list_batches(self, store: PersistentStore) -> None:
        store.save_batch("batch-1", {"id": "batch-1", "status": "pending"})
        store.save_batch("batch-2", {"id": "batch-2", "status": "completed"})
        batches = store.list_batches()
        assert len(batches) == 2
        # Newest first (ORDER BY created_at DESC)
        assert batches[0]["id"] == "batch-2"

    def test_list_batches_with_limit(self, store: PersistentStore) -> None:
        for i in range(5):
            store.save_batch(f"batch-{i}", {"id": f"batch-{i}"})
        assert len(store.list_batches(limit=3)) == 3

    def test_list_batches_with_after_cursor(self, store: PersistentStore) -> None:
        """Cursor-based pagination via the ``after`` parameter."""
        store.save_batch("a", {"id": "a"})
        store.save_batch("b", {"id": "b"})
        store.save_batch("c", {"id": "c"})
        batches = store.list_batches(after="c")
        assert len(batches) == 2

    def test_update_batch(self, store: PersistentStore) -> None:
        store.save_batch("batch-1", {"id": "batch-1", "status": "pending"})
        updated = store.update_batch("batch-1", {"status": "completed"})
        assert updated["status"] == "completed"
        # Verify persistence
        result = store.get_batch("batch-1")
        assert result["status"] == "completed"

    def test_update_batch_merges_fields(self, store: PersistentStore) -> None:
        store.save_batch("batch-1", {"id": "batch-1", "a": 1, "b": 2})
        store.update_batch("batch-1", {"b": 99, "c": 3})
        result = store.get_batch("batch-1")
        assert result["a"] == 1
        assert result["b"] == 99
        assert result["c"] == 3

    def test_update_missing_batch(self, store: PersistentStore) -> None:
        assert store.update_batch("nonexistent", {"status": "done"}) is None

    def test_delete_batch(self, store: PersistentStore) -> None:
        store.save_batch("batch-1", {"id": "batch-1"})
        assert store.delete_batch("batch-1") is True
        assert store.get_batch("batch-1") is None

    def test_delete_missing_batch(self, store: PersistentStore) -> None:
        assert store.delete_batch("nonexistent") is False

    def test_double_delete_batch(self, store: PersistentStore) -> None:
        """Second delete returns False."""
        store.save_batch("batch-1", {"id": "batch-1"})
        assert store.delete_batch("batch-1") is True
        assert store.delete_batch("batch-1") is False

    def test_save_batch_replaces_existing(self, store: PersistentStore) -> None:
        store.save_batch("batch-1", {"id": "batch-1", "status": "old"})
        store.save_batch("batch-1", {"id": "batch-1", "status": "new"})
        assert store.get_batch("batch-1")["status"] == "new"

    def test_save_batch_with_custom_timestamp(self, store: PersistentStore) -> None:
        """Explicit ``created_at`` in data dict is stored as the row timestamp."""
        ts1 = 1000.0
        ts2 = 2000.0
        store.save_batch("old-batch", {"id": "old-batch", "created_at": ts1})
        store.save_batch("new-batch", {"id": "new-batch", "created_at": ts2})
        batches = store.list_batches()
        assert batches[0]["id"] == "new-batch"


# ======================================================================
# File CRUD
# ======================================================================


class TestFileCRUD:
    def test_save_and_get_file(self, store: PersistentStore) -> None:
        store.save_file(
            "file-1",
            {"id": "file-1", "filename": "test.json", "purpose": "fine-tune"},
        )
        result = store.get_file("file-1")
        assert result is not None
        assert result["filename"] == "test.json"

    def test_get_missing_file(self, store: PersistentStore) -> None:
        assert store.get_file("nonexistent") is None

    def test_list_files_empty(self, store: PersistentStore) -> None:
        assert store.list_files() == []

    def test_list_files(self, store: PersistentStore) -> None:
        store.save_file("file-1", {"id": "file-1", "purpose": "fine-tune"})
        store.save_file("file-2", {"id": "file-2", "purpose": "batch"})
        files = store.list_files()
        assert len(files) == 2

    def test_list_files_filter_by_purpose(self, store: PersistentStore) -> None:
        store.save_file("file-1", {"id": "file-1", "purpose": "fine-tune"})
        store.save_file("file-2", {"id": "file-2", "purpose": "batch"})
        store.save_file("file-3", {"id": "file-3", "purpose": "fine-tune"})
        files = store.list_files(purpose="fine-tune")
        assert len(files) == 2

    def test_list_files_filter_no_match(self, store: PersistentStore) -> None:
        """Purpose filter with no matches returns empty list."""
        store.save_file("file-1", {"id": "file-1", "purpose": "batch"})
        assert store.list_files(purpose="nonexistent") == []

    def test_delete_file(self, store: PersistentStore) -> None:
        store.save_file("file-1", {"id": "file-1"})
        assert store.delete_file("file-1") is True
        assert store.get_file("file-1") is None

    def test_delete_missing_file(self, store: PersistentStore) -> None:
        assert store.delete_file("nonexistent") is False

    def test_double_delete_file(self, store: PersistentStore) -> None:
        """Second delete returns False."""
        store.save_file("file-1", {"id": "file-1"})
        assert store.delete_file("file-1") is True
        assert store.delete_file("file-1") is False

    def test_save_file_replaces_existing(self, store: PersistentStore) -> None:
        store.save_file("file-1", {"id": "file-1", "status": "old"})
        store.save_file("file-1", {"id": "file-1", "status": "new"})
        assert store.get_file("file-1")["status"] == "new"


# ======================================================================
# Fine-tuning job CRUD
# ======================================================================


class TestFineTuningCRUD:
    def test_save_and_get_job(self, store: PersistentStore) -> None:
        store.save_fine_tuning_job(
            "ft-1", {"id": "ft-1", "model": "test-model", "status": "queued"}
        )
        result = store.get_fine_tuning_job("ft-1")
        assert result["model"] == "test-model"
        assert result["status"] == "queued"

    def test_get_missing_job(self, store: PersistentStore) -> None:
        assert store.get_fine_tuning_job("nonexistent") is None

    def test_list_jobs_empty(self, store: PersistentStore) -> None:
        assert store.list_fine_tuning_jobs() == []

    def test_list_jobs(self, store: PersistentStore) -> None:
        store.save_fine_tuning_job("ft-1", {"id": "ft-1"})
        store.save_fine_tuning_job("ft-2", {"id": "ft-2"})
        jobs = store.list_fine_tuning_jobs()
        assert len(jobs) == 2

    def test_list_jobs_with_limit(self, store: PersistentStore) -> None:
        for i in range(5):
            store.save_fine_tuning_job(f"ft-{i}", {"id": f"ft-{i}"})
        assert len(store.list_fine_tuning_jobs(limit=3)) == 3

    def test_update_job(self, store: PersistentStore) -> None:
        store.save_fine_tuning_job(
            "ft-1", {"id": "ft-1", "status": "queued"}
        )
        updated = store.update_fine_tuning_job("ft-1", {"status": "running"})
        assert updated["status"] == "running"

    def test_update_job_merges_fields(self, store: PersistentStore) -> None:
        store.save_fine_tuning_job(
            "ft-1", {"id": "ft-1", "model": "base", "status": "queued"}
        )
        store.update_fine_tuning_job(
            "ft-1", {"status": "running", "progress": 0.5}
        )
        result = store.get_fine_tuning_job("ft-1")
        assert result["model"] == "base"
        assert result["status"] == "running"
        assert result["progress"] == 0.5

    def test_update_missing_job(self, store: PersistentStore) -> None:
        assert (
            store.update_fine_tuning_job("nonexistent", {"status": "done"})
            is None
        )


# ======================================================================
# Schema migration
# ======================================================================


class TestSchemaMigration:
    def test_schema_version_initialized(self, store: PersistentStore) -> None:
        """Fresh ``:memory:`` database is at ``SCHEMA_VERSION``."""
        assert store._get_schema_version() == store.SCHEMA_VERSION

    def test_set_schema_version_updates_value(self, store: PersistentStore) -> None:
        """``_set_schema_version`` stores and ``_get_schema_version`` retrieves."""
        store._set_schema_version(42)
        assert store._get_schema_version() == 42

    def test_set_schema_version_overwrites(self, store: PersistentStore) -> None:
        """Repeated calls overwrite the stored version."""
        store._set_schema_version(1)
        store._set_schema_version(2)
        assert store._get_schema_version() == 2

    def test_migrate_does_not_downgrade(self, store: PersistentStore) -> None:
        """``_migrate`` is a no-op when stored version exceeds ``SCHEMA_VERSION``."""
        store._set_schema_version(999)
        store._migrate()
        assert store._get_schema_version() == 999

    def test_migrate_is_idempotent(self, store: PersistentStore) -> None:
        """Calling ``_migrate`` multiple times leaves version unchanged."""
        assert store._get_schema_version() == store.SCHEMA_VERSION
        store._migrate()
        assert store._get_schema_version() == store.SCHEMA_VERSION

    def test_empty_schema_table_returns_zero(self, store: PersistentStore) -> None:
        """``_get_schema_version`` returns 0 when the version table is empty."""
        with store._transaction(write=True) as conn:
            conn.execute("DELETE FROM schema_version")
        assert store._get_schema_version() == 0

    def test_init_new_db_creates_tables(self) -> None:
        """A brand-new store creates all expected tables."""
        s = PersistentStore(db_path=":memory:")
        with s._transaction() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        assert "batches" in tables
        assert "files" in tables
        assert "fine_tuning_jobs" in tables
        assert "schema_version" in tables

    def test_init_new_db_creates_indexes(self) -> None:
        """Tables are created with the expected indexes."""
        s = PersistentStore(db_path=":memory:")
        with s._transaction() as conn:
            indexes = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            ]
        assert "idx_batches_created" in indexes
        assert "idx_files_created" in indexes
        assert "idx_ft_jobs_created" in indexes


# ======================================================================
# Thread safety
# ======================================================================


class TestThreadSafety:
    def test_locks_exist(self, store: PersistentStore) -> None:
        """The store has separate read and write locks."""
        assert store._lock is not None
        assert store._write_lock is not None

    def test_concurrent_writes_do_not_corrupt(
        self, file_store: PersistentStore
    ) -> None:
        """20 concurrent ``save_batch`` calls all succeed without error."""
        errors: list[tuple[int, BaseException]] = []
        lock = threading.Lock()

        def writer(i: int) -> None:
            try:
                file_store.save_batch(
                    f"batch-{i}", {"id": f"batch-{i}", "num": i}
                )
            except Exception as e:
                with lock:
                    errors.append((i, e))

        threads = [
            threading.Thread(target=writer, args=(i,)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent write errors: {errors}"

        # All batches are retrievable
        for i in range(20):
            assert (
                file_store.get_batch(f"batch-{i}") is not None
            ), f"batch-{i} missing after concurrent writes"

    def test_concurrent_reads_during_writes(
        self, file_store: PersistentStore
    ) -> None:
        """Concurrent readers do not fail while a writer updates the same row."""
        file_store.save_batch(
            "target", {"id": "target", "value": "original"}
        )
        errors: list[BaseException] = []
        lock = threading.Lock()

        def reader() -> None:
            for _ in range(50):
                try:
                    file_store.get_batch("target")
                except Exception as e:
                    with lock:
                        errors.append(e)

        def writer() -> None:
            for i in range(50):
                file_store.save_batch(
                    "target", {"id": "target", "value": f"update-{i}"}
                )

        threads: list[threading.Thread] = [
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_mixed_operations(
        self, file_store: PersistentStore
    ) -> None:
        """Concurrent saves, gets, updates, and deletes do not raise."""
        for i in range(10):
            file_store.save_batch(f"item-{i}", {"id": f"item-{i}"})

        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(n: int) -> None:
            try:
                for _ in range(20):
                    idx = n % 10
                    file_store.get_batch(f"item-{idx}")
                    file_store.update_batch(
                        f"item-{idx}", {"visited": True}
                    )
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_writer_exclusion(self, file_store: PersistentStore) -> None:
        """The write lock cannot be acquired reentrantly by a second thread."""
        acquired = threading.Event()
        done = threading.Event()
        errors: list[BaseException] = []

        def hold_write_lock() -> None:
            try:
                with file_store._write_lock:
                    acquired.set()
                    done.wait(timeout=5)
            except Exception as e:
                errors.append(e)

        holder = threading.Thread(target=hold_write_lock, daemon=True)
        holder.start()
        acquired.wait(timeout=5)

        # Second thread attempting the same lock should NOT succeed
        second_acquired = threading.Event()

        def try_write_lock() -> None:
            ok = file_store._write_lock.acquire(blocking=False)
            if ok:
                second_acquired.set()
                file_store._write_lock.release()

        contender = threading.Thread(target=try_write_lock, daemon=True)
        contender.start()
        contender.join(timeout=2)
        assert (
            not second_acquired.is_set()
        ), "Second thread acquired the write lock while holder held it"
        done.set()
        holder.join(timeout=2)
        assert len(errors) == 0


# ======================================================================
# Connection pool
# ======================================================================


class TestConnectionPool:
    def test_pool_recycles_connections(
        self, file_store: PersistentStore
    ) -> None:
        """Returned connection is reused on next ``_get_conn``."""
        conn1 = file_store._get_conn()
        file_store._return_conn(conn1)
        conn2 = file_store._get_conn()
        assert conn1 is conn2
        file_store._return_conn(conn2)

    def test_pool_excess_connections_closed(
        self, file_store: PersistentStore
    ) -> None:
        """Returning more than ``pool_size`` connections closes extras."""
        conns = [
            file_store._get_conn()
            for _ in range(file_store._pool_size + 2)
        ]
        for c in conns:
            file_store._return_conn(c)
        assert len(file_store._pool) <= file_store._pool_size

    def test_pool_default_size(self) -> None:
        """Default ``pool_size`` is 4."""
        s = PersistentStore(db_path=":memory:")
        assert s._pool_size == 4

    def test_pool_custom_size(self) -> None:
        """``pool_size`` can be overridden in the constructor."""
        s = PersistentStore(db_path=":memory:", pool_size=2)
        assert s._pool_size == 2

    def test_pool_connections_are_functional_after_return(
        self, file_store: PersistentStore
    ) -> None:
        """A connection retrieved from the pool can execute queries."""
        conn = file_store._get_conn()
        file_store._return_conn(conn)
        conn2 = file_store._get_conn()
        row = conn2.execute("SELECT 1 AS val").fetchone()
        assert row["val"] == 1
        file_store._return_conn(conn2)

    def test_concurrent_pool_access(
        self, file_store: PersistentStore
    ) -> None:
        """Multiple threads cycle connections without error."""
        errors: list[BaseException] = []
        lock = threading.Lock()

        def cycle() -> None:
            try:
                for _ in range(10):
                    conn = file_store._get_conn()
                    conn.execute("SELECT 1")
                    file_store._return_conn(conn)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=cycle) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_pool_does_not_exceed_capacity(
        self, file_store: PersistentStore
    ) -> None:
        """Pool never grows beyond ``pool_size`` after multiple cycles."""
        # Take all connections from the pool so it empties.
        taken: list[sqlite3.Connection] = []
        for _ in range(file_store._pool_size):
            taken.append(file_store._get_conn())
        assert len(file_store._pool) == 0

        # Return them all -- pool refills to pool_size.
        for c in taken:
            file_store._return_conn(c)
        assert len(file_store._pool) == file_store._pool_size

        # Return extra connections -- they should be closed, not added.
        extras = [file_store._get_conn() for _ in range(3)]
        for c in extras:
            file_store._return_conn(c)
        assert len(file_store._pool) == file_store._pool_size

        # Close extra connections to avoid ResourceWarning.
        for c in extras:
            try:
                c.close()
            except Exception:
                pass


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_large_json_data(self, store: PersistentStore) -> None:
        """Large JSON payloads (10k chars) round-trip correctly."""
        big_data = {"data": "x" * 10000}
        store.save_batch("big", big_data)
        result = store.get_batch("big")
        assert result is not None
        assert result["data"] == big_data["data"]

    def test_special_characters(self, store: PersistentStore) -> None:
        """Unicode characters survive the serialization round-trip."""
        data = {"text": "heallo world \U0001f525"}
        store.save_batch("unicode", data)
        result = store.get_batch("unicode")
        assert result["text"] == "heallo world \U0001f525"

    def test_empty_dict_data(self, store: PersistentStore) -> None:
        """Saving and retrieving an empty dict works."""
        store.save_batch("empty", {})
        result = store.get_batch("empty")
        assert result is not None
        assert result == {}

    def test_nested_json(self, store: PersistentStore) -> None:
        """Nested dicts and lists are preserved."""
        data = {"nested": {"deep": [1, 2, 3], "key": "value"}}
        store.save_batch("nested", data)
        result = store.get_batch("nested")
        assert result["nested"]["deep"] == [1, 2, 3]

    def test_null_values(self, store: PersistentStore) -> None:
        """Python ``None`` values are preserved."""
        data = {"id": None, "name": None, "tags": [None]}
        store.save_batch("nulls", data)
        result = store.get_batch("nulls")
        assert result["id"] is None
        assert result["name"] is None
        assert result["tags"] == [None]

    def test_numeric_types(self, store: PersistentStore) -> None:
        """Integers, floats, and booleans round-trip correctly."""
        data = {
            "int": 42,
            "float": 3.14,
            "bool_true": True,
            "bool_false": False,
        }
        store.save_batch("nums", data)
        result = store.get_batch("nums")
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool_true"] is True
        assert result["bool_false"] is False

    def test_list_data(self, store: PersistentStore) -> None:
        """Top-level lists (stored as JSON inside the dict) round-trip."""
        data = {"items": [1, 2, 3, 4, 5]}
        store.save_batch("list-data", data)
        result = store.get_batch("list-data")
        assert result["items"] == [1, 2, 3, 4, 5]

    def test_very_deeply_nested_data(self, store: PersistentStore) -> None:
        """Deeply nested JSON survives serialization."""
        inner: dict = {}
        for _ in range(50):
            inner = {"level": inner}
        data = {"deep": inner}
        store.save_batch("deep-nest", data)
        result = store.get_batch("deep-nest")
        assert result is not None

    def test_get_after_delete_returns_none(self, store: PersistentStore) -> None:
        """``get_batch`` returns ``None`` for a deleted batch."""
        store.save_batch("temp", {"id": "temp"})
        store.delete_batch("temp")
        assert store.get_batch("temp") is None

    def test_list_after_delete(self, store: PersistentStore) -> None:
        """Deleted items no longer appear in ``list_batches``."""
        store.save_batch("a", {"id": "a"})
        store.save_batch("b", {"id": "b"})
        store.delete_batch("a")
        batches = store.list_batches()
        ids = [b["id"] for b in batches]
        assert "a" not in ids
        assert "b" in ids

    def test_empty_string_values(self, store: PersistentStore) -> None:
        """Empty strings are preserved."""
        data = {"id": "", "name": "valid"}
        store.save_batch("empty-str", data)
        result = store.get_batch("empty-str")
        assert result["id"] == ""

    def test_boolean_false_values(self, store: PersistentStore) -> None:
        """``False`` boolean values are preserved (common JSON pitfall)."""
        data = {"active": False, "visible": False}
        store.save_batch("false-vals", data)
        result = store.get_batch("false-vals")
        assert result["active"] is False
        assert result["visible"] is False

    def test_zero_numeric_values(self, store: PersistentStore) -> None:
        """Zero numeric values are preserved."""
        data = {"count": 0, "ratio": 0.0}
        store.save_batch("zeros", data)
        result = store.get_batch("zeros")
        assert result["count"] == 0
        assert result["ratio"] == 0.0

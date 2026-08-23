"""Benchmark: SQLite persistent store throughput.

Measures operations per second with and without the connection pool.

Target: Pooled connections should show higher throughput under batch writes.
"""

from __future__ import annotations

import pytest
from distllm.api.persistent_store import PersistentStore


class TestSQLiteBatchThroughput:
    """Measure SQLite write throughput under batch load."""

    def test_batch_write_throughput(self, benchmark):
        """Write 100 batches and measure ops/s."""
        store = PersistentStore(db_path=":memory:")

        def _write_batches():
            for i in range(100):
                store.save_batch(f"perf-batch-{i}", {"id": f"perf-{i}", "data": "x" * 1000})

        benchmark(_write_batches)

    def test_concurrent_read_write(self, benchmark):
        """Concurrent reads and writes under load."""
        import threading
        store = PersistentStore(db_path=":memory:")
        for i in range(50):
            store.save_batch(f"seed-{i}", {"id": f"seed-{i}"})

        def _mixed():
            def writer(n: int):
                for i in range(10):
                    store.save_batch(f"load-{n}-{i}", {"id": f"load-{n}-{i}"})

            def reader():
                for _ in range(10):
                    store.list_batches()

            threads = []
            for i in range(4):
                t = threading.Thread(target=writer, args=(i,))
                threads.append(t)
            for _ in range(4):
                t = threading.Thread(target=reader)
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        benchmark(_mixed)

    def test_large_json_throughput(self, benchmark):
        """Throughput with large JSON payloads (10KB)."""
        store = PersistentStore(db_path=":memory:")
        big_data = {"data": "x" * 10000}

        def _write_large():
            for i in range(50):
                store.save_batch(f"large-{i}", big_data)

        # Number of operations per second
        result = benchmark(_write_large)
        if result is not None:
            pass  # pytest-benchmark tracks the stats

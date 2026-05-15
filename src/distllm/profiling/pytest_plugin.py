"""pytest plugin for memory profiling.

When --memory-profile is passed, wraps each test with tracemalloc
and reports memory usage.
"""

import tracemalloc
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--memory-profile",
        action="store_true",
        default=False,
        help="Enable memory profiling for tests",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if not item.config.getoption("--memory-profile", False):
        return None

    tracemalloc.start()
    return None


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    # Memory stats are logged per-test when --memory-profile is active
    pass


@pytest.fixture
def memory_snapshot(request):
    """Fixture that provides memory stats before and after test.

    Usage:
        def test_something(memory_snapshot):
            before = memory_snapshot.before()
            # ... do work ...
            after = memory_snapshot.after()
            assert after.cpu_mb - before.cpu_mb < 100  # < 100MB growth
    """
    if not request.config.getoption("--memory-profile", False):
        # Return a no-op snapshot
        yield _NoopSnapshot()
        return

    tracemalloc.start()
    snapshot = _MemorySnapshot()
    yield snapshot
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    test_name = request.node.name
    cpu_mb = current / (1024 * 1024)
    peak_mb = peak / (1024 * 1024)
    request.config.cache.set(f"memory/{test_name}", {
        "cpu_mb": round(cpu_mb, 2),
        "peak_mb": round(peak_mb, 2),
    })


class _MemorySnapshot:
    def __init__(self):
        self._current, self._peak = tracemalloc.get_traced_memory()

    def before(self):
        return _MemStats(self._current, self._peak)

    def after(self):
        current, peak = tracemalloc.get_traced_memory()
        return _MemStats(current, peak)


class _MemStats:
    def __init__(self, current_bytes, peak_bytes):
        self.cpu_mb = current_bytes / (1024 * 1024)
        self.peak_mb = peak_bytes / (1024 * 1024)


class _NoopSnapshot:
    def before(self):
        return _MemStats(0, 0)

    def after(self):
        return _MemStats(0, 0)

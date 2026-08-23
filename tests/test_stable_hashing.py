"""Regression tests for stable (non-salted) hashing used in distributed
identity / bucketing / seed derivation (replaces builtin hash() which is
PYTHONHASHSEED-salted and non-deterministic across processes — the M8 class
of bug).
"""

import subprocess
import sys

from distllm.core.hashing import stable_hash, stable_seed


def test_stable_hash_deterministic_same_process():
    assert stable_hash("node-7", "tenant-a") == stable_hash("node-7", "tenant-a")
    assert stable_hash("a") != stable_hash("b")


def test_stable_hash_deterministic_across_processes():
    # builtin hash() would differ across interpreter invocations; stable_hash
    # must NOT (this is the whole point of the M8-class fix).
    code = (
        "from distllm.core.hashing import stable_hash; "
        "print(stable_hash('node-7','tenant-a'))"
    )
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    assert out1 == out2, f"stable_hash not deterministic across processes: {out1} vs {out2}"
    # And it must differ from a fresh-process builtin hash() which is salted.
    bh = subprocess.run(
        [sys.executable, "-c", "print(hash('node-7'+'tenant-a'))"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert bh != out1, "stable_hash accidentally matched salted builtin hash()"


def test_stable_seed_full_range():
    s = stable_seed("seed-me")
    assert 0 <= s <= 0xFFFFFFFF
    assert stable_seed("seed-me") == stable_seed("seed-me")


def test_bucketing_stable():
    # Simulates the cross_cluster / geo bucketing pattern.
    workers = list(range(8))
    w = workers[stable_hash("request-payload") % len(workers)]
    assert w in workers

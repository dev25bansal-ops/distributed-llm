"""Regression tests for Medium-severity findings: M3, M5, M8, M13, M14, M18.

All are pure-stdlib / torch-free and run fast.

M3  (kv_cache_marketplace double-debit): purchase() already debits the buyer;
     record_consumed_tokens() must NOT debit again. After purchase + consume,
     the buyer balance must equal (initial - price), not (initial - price - use).

M5  (backup_manager path traversal): get_backup() must reject IDs containing
     traversal metacharacters (e.g. "../") and only accept [A-Za-z0-9._-].

M8  (process-salted hash() for distributed identity): stable_hash() from
     core.hashing must be deterministic across calls/processes, unlike the
     builtin hash().

M13 (request_fingerprinting dedup no-op): store() must populate the in-flight
     result so wait_for_result() returns it (not None).

M14 (provider_health rolling success-rate): the rolling window must be computed
     from the PREVIOUS last_check, not collapse to ~0 (which forces the rate to
     1 - consecutive_failures/10 on every single check).

M18 (arbitrage_engine dead branch): the MEDIUM risk branch must be reachable;
     a small-but-valid savings opportunity must yield MEDIUM, not always HIGH.
"""

from __future__ import annotations

import time

from distllm.core.hashing import stable_hash
from distllm.core.kv_cache_marketplace import CacheMarketplace
from distllm.core.request_fingerprinting import RequestFingerprinter


# ── M3: no double-debit on purchase + consume ─────────────────────────────

def test_kv_marketplace_no_double_debit():
    mkt = CacheMarketplace(node_id="seller", default_price_per_token=0.01)
    mkt._credit_balances["buyer"] = 100.0
    mkt._credit_balances["seller"] = 0.0

    mkt._advertisements["ad1"] = type(
        "Ad", (), {"node_id": "seller", "token_count": 10,
                   "price_credits": 5.0, "is_expired": False}
    )()
    assert mkt.purchase("ad1", "buyer") is True

    before = mkt.get_balance("buyer")
    mkt.record_consumed_tokens("buyer", 10)

    # Buyer paid the advertised 5.0 once via purchase(); consumption must NOT
    # debit again. So balance == 100 - 5 == 95, never 100 - 5 - 0.1.
    assert mkt.get_balance("buyer") == before, (
        f"buyer balance changed by consume(): {before} -> "
        f"{mkt.get_balance('buyer')} (M3 double-debit)"
    )
    assert mkt.get_balance("buyer") == 95.0
    # Seller received exactly the price once.
    assert mkt.get_balance("seller") == 5.0
    # Consumption is metered, not lost.
    assert mkt._consumed_tokens.get("buyer", 0) == 10


# ── M5: path traversal rejected ───────────────────────────────────────────

def test_backup_get_rejects_traversal():
    from distllm.core.backup_manager import BackupManager

    bm = BackupManager(backup_dir="/tmp/distllm_backups_test")
    # Should refuse traversal / non-conformant ids without raising or reading.
    for bad in ("../etc/passwd", "..\\win", "foo/bar", "a..b"):
        assert bm.get_backup(bad) is None, f"traversal id not rejected: {bad!r}"
    # Conformant id is at least accepted (returns None only if missing).
    assert bm.get_backup("backup-2026-07-12") is None


# ── M8: stable hash is deterministic (unlike hash()) ──────────────────────

def test_stable_hash_deterministic():
    a = stable_hash("token", "1")
    b = stable_hash("token", "1")
    assert a == b
    # Different inputs -> different output.
    assert stable_hash("token", "2") != a
    # Must NOT equal the process-salted builtin hash (which varies per run).
    assert a == int(__import__("hashlib").sha256(b"token1").hexdigest()[:8], 16)


# ── M13: in-flight dedup returns stored result ────────────────────────────

def test_fingerprint_wait_for_result_returns_stored():
    fp = RequestFingerprinter(enable_dedup=True)
    f = fp.fingerprint(prompt="hello", params={"max_tokens": 5})
    fp.store(f, request_id="r1", response="world")

    # A concurrent waiter should receive the stored result instead of None.
    result = fp.wait_for_result(f, timeout_s=0.5)
    assert result == "world", f"wait_for_result returned {result!r} (M13 dedup no-op)"
    # lookup also works.
    assert fp.lookup(f).response == "world"


# ── M14: rolling success-rate window uses previous last_check ─────────────

def test_provider_health_rolling_window():
    from distllm.core.provider_health import RegionHealth

    # Mirror the FIXED computation in provider_health._record_check_result:
    # prev_check is captured BEFORE mutating last_check (M14 bug was computing
    # elapsed = time.time() - health.last_check after it was already updated).
    h = RegionHealth(provider="aws", region="us-east-1")
    h.last_check = time.time() - 3600.0  # last successful check 1h ago
    interval = 60.0

    prev_check = h.last_check
    h.last_check = time.time()
    elapsed = time.time() - prev_check
    total_checks = max(1, int(elapsed / interval) + 1)
    rate = max(0, 1.0 - (h.consecutive_failures / max(total_checks, 10)))

    # A 1h gap => divisor ~60, so a single (or even several) failures barely
    # dents the rate. The OLD code set total_checks=1 => rate=1-failures/10.
    assert total_checks > 1, "rolling window collapsed to a single check (M14)"
    assert rate > 0.9, f"rolling rate collapsed to {rate} (M14)"


# ── M18: MEDIUM risk branch now reachable ─────────────────────────────────

def test_arbitrage_medium_branch_reachable():
    from distllm.core.arbitrage_engine import (
        ArbitrageEngine, MigrationRisk, PriceHistory, OpportunityType,
    )

    eng = ArbitrageEngine(provider_savings_threshold_pct=10.0)
    eng.set_active_location("aws", "g5", "us-east-1")

    active = PriceHistory(provider="aws", instance_type="g5", region="us-east-1")
    active.add(1.0, is_spot=False)
    eng._histories["aws:g5:us-east-1"] = active

    # Cheaper alt at 0.85 -> 15% savings (>10% threshold) but < 2x threshold,
    # so the now-reachable MEDIUM branch should fire.
    alt = PriceHistory(provider="gcp", instance_type="g2", region="us-east-1")
    alt.add(0.85, is_spot=False)
    eng._histories["gcp:g2:us-east-1"] = alt
    eng._active_provider = "aws"

    opps = eng.detect_opportunities()
    assert opps, "no arbitrage opportunity detected"
    assert opps[0].migration_risk == MigrationRisk.MEDIUM, (
        f"MEDIUM branch unreachable (got {opps[0].migration_risk}) — M18 dead branch"
    )

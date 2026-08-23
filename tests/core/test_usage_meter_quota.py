"""Tests for max_tokens_per_request quota enforcement in UsageMeter."""

from distllm.core.usage_meter import QuotaLimit, UsageMeter


def _meter() -> UsageMeter:
    return UsageMeter(use_sqlite=False)


def test_per_request_limit_blocks_oversize_request():
    meter = _meter()
    meter.set_quota(
        "tenant-1",
        QuotaLimit(tenant_id="tenant-1", max_tokens_per_request=1000),
    )

    allowed, reason = meter.check_quota("tenant-1", requested_tokens=2000)
    assert allowed is False
    assert "per-request token limit" in reason


def test_per_request_limit_allows_within_limit():
    meter = _meter()
    meter.set_quota(
        "tenant-1",
        QuotaLimit(tenant_id="tenant-1", max_tokens_per_request=1000),
    )

    allowed, _ = meter.check_quota("tenant-1", requested_tokens=500)
    assert allowed is True


def test_per_request_limit_skipped_when_unset():
    meter = _meter()
    meter.set_quota(
        "tenant-1",
        QuotaLimit(tenant_id="tenant-1", max_tokens_per_request=0),
    )

    allowed, _ = meter.check_quota("tenant-1", requested_tokens=10_000)
    assert allowed is True


def test_enforce_quota_blocks_oversize_request():
    meter = _meter()
    meter.set_quota(
        "tenant-1",
        QuotaLimit(tenant_id="tenant-1", max_tokens_per_request=1000),
    )

    allowed, reason = meter.enforce_quota("tenant-1", requested_tokens=2000)
    assert allowed is False
    assert "per-request token limit" in reason
    # Must not have incremented concurrency for a blocked request.
    assert meter.get_concurrent("tenant-1") == 0

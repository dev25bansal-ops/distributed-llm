"""Regression test for Money-backed cost accumulation in TenantCostAttribution.

Proves the systemic float-drift risk (M1 class) is fixed at the accumulation
site: repeatedly summing many small costs into the hourly/daily/monthly
ledgers stays exact via Decimal-backed Money, and the public header API still
returns float strings unchanged.
"""

from distllm.core.tenant_cost_attribution import TenantCostAttribution


def _record_many(attr, n=10000, unit=0.0001):
    """Record n requests each costing `unit` USD."""
    for i in range(n):
        attr.record(
            tenant_id="t",
            request_id=f"r{i}",
            estimated_cost_usd=unit,
            actual_cost_usd=unit,
        )


def test_hourly_accumulation_is_exact():
    attr = TenantCostAttribution()
    _record_many(attr, n=10000, unit=0.0001)
    # 10000 * 0.0001 = 1.0000 USD exactly (float would drift).
    headers = attr.get_cost_headers("t", attr._records["t"][-1])
    hourly = float(headers["X-DistLLM-Tenant-Hourly-Cost"])
    daily = float(headers["X-DistLLM-Tenant-Daily-Cost"])
    # Allow 1e-9 tolerance for the final float conversion; exact Decimal sum.
    assert abs(hourly - 1.0) < 1e-9, f"hourly drifted to {hourly}"
    assert abs(daily - 1.0) < 1e-9, f"daily drifted to {daily}"


def test_accumulator_matches_decimal_sum():
    attr = TenantCostAttribution()
    _record_many(attr, n=7000, unit=0.000123)
    # expected = 7000 * 0.000123 = 0.861 (exact in Decimal; float would drift)
    # Read the EXACT accumulator value (not the cent-quantized display view).
    from decimal import Decimal

    exact = attr._hourly_costs["t"].value()
    assert exact == Decimal("0.861"), f"exact accumulator = {exact}, expected 0.861"

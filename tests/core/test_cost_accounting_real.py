"""Real-object cost-accounting tests — no test doubles.

Exercises the actual ``UsageMeter`` against real in-memory records: token
accounting, per-request cost from the configured prices, tenant aggregation,
and invoice generation.
"""

from __future__ import annotations

from distllm.core.usage_meter import UsageMeter


class TestRealUsageMeter:
    def test_token_and_cost_accounting(self, tmp_path) -> None:
        meter = UsageMeter(
            storage_path=str(tmp_path / "usage.jsonl"),
            input_price=0.001, output_price=0.002, use_sqlite=False,
        )
        meter.record_request(
            "tenant-a", "model-x", input_tokens=100, output_tokens=50,
            duration_ms=10.0, endpoint="/v1/chat", key_id="k1",
        )
        meter.record_request(
            "tenant-a", "model-x", input_tokens=200, output_tokens=30,
            duration_ms=12.0, endpoint="/v1/chat", key_id="k1",
        )

        total = meter.total_usage()
        assert total["total_requests"] == 2
        assert total["total_input_tokens"] == 300
        assert total["total_output_tokens"] == 80
        # cost = (300/1000 * 0.001) + (80/1000 * 0.002) = 0.0003 + 0.00016,
        # rounded to 4 decimal places by total_usage()
        assert total["total_cost"] == round(0.00046, 4)

    def test_tenant_isolation_in_per_tenant_summary(self, tmp_path) -> None:
        meter = UsageMeter(
            storage_path=str(tmp_path / "usage.jsonl"),
            input_price=0.001, output_price=0.002, use_sqlite=False,
        )
        meter.record_request("tenant-a", "m", input_tokens=100, output_tokens=10,
                             endpoint="/v1/chat")
        meter.record_request("tenant-b", "m", input_tokens=50, output_tokens=20,
                             endpoint="/v1/chat")

        summary = {r.tenant_id: r for r in meter._tenants.values()}
        assert summary["tenant-a"].total_input_tokens == 100
        assert summary["tenant-b"].total_input_tokens == 50

    def test_invoice_covers_records(self, tmp_path) -> None:
        meter = UsageMeter(
            storage_path=str(tmp_path / "usage.jsonl"),
            use_sqlite=False,
        )
        meter.record_request("tenant-x", "m", input_tokens=10, output_tokens=10,
                             endpoint="/v1/chat")
        invoice = meter.generate_invoice(tenant_id="tenant-x", period_start=0.0)
        assert invoice.get("total_requests") == 1
        assert invoice.get("total_input_tokens") == 10
        assert invoice.get("total_output_tokens") == 10
        assert invoice.get("grand_total") == invoice.get("total_cost")
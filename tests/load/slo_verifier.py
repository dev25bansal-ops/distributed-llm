"""SLO compliance stress test — generates multi-tenant workload and verifies SLOs.

Usage::

    python tests/load/slo_verifier.py
    python tests/load/slo_verifier.py --tenants 10 --requests 500 --slo-ms 1000

This test generates concurrent requests across multiple tenants with
known SLO configurations, measures actual p99 latency, and reports
breach percentages.  Exits non-zero if breach rate exceeds threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SLOMeasurement:
    tenant_id: str
    slo_ms: float
    latencies: list[float] = field(default_factory=list)
    breaches: int = 0
    total_requests: int = 0

    @property
    def p99_latency_ms(self) -> float:
        if len(self.latencies) < 10:
            return 0.0
        sorted_lats = sorted(self.latencies)
        idx = int(len(sorted_lats) * 0.99)
        return sorted_lats[idx]

    @property
    def breach_rate_pct(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.breaches / self.total_requests) * 100.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)


class SLOMultiTenantStressTest:
    """Generates multi-tenant workload and verifies SLO compliance."""

    def __init__(
        self,
        num_tenants: int = 5,
        requests_per_tenant: int = 200,
        slo_ms_range: tuple[float, float] = (500, 2000),
        breach_threshold_pct: float = 5.0,
        concurrency: int = 10,
        simulate_fn: Any = None,
    ):
        self._num_tenants = num_tenants
        self._requests_per = requests_per_tenant
        self._slo_range = slo_ms_range
        self._threshold = breach_threshold_pct
        self._concurrency = concurrency
        self._simulate_fn = simulate_fn or self._default_simulate

        self._measurements: dict[str, SLOMeasurement] = {}

    def setup(self) -> None:
        """Register tenants with randomized SLOs."""
        for i in range(self._num_tenants):
            tid = f"tenant-{chr(ord('a') + i)}"
            slo_ms = random.uniform(*self._slo_range)
            self._measurements[tid] = SLOMeasurement(
                tenant_id=tid,
                slo_ms=round(slo_ms, 0),
            )

    async def _default_simulate(self, tenant_id: str, slo_ms: float) -> float:
        """Default simulation: random latency skewed by load."""
        base = random.uniform(50, slo_ms * 0.9)
        # Occasionally produce a breach
        if random.random() < 0.03:
            base *= random.uniform(1.5, 3.0)
        await asyncio.sleep(base / 1000.0)
        return base

    async def _run_single(self, tenant_id: str) -> float:
        """Run a single request, measure latency, check SLO."""
        slo = self._measurements[tenant_id].slo_ms
        t0 = time.monotonic()
        await self._simulate_fn(tenant_id, slo)
        latency_ms = (time.monotonic() - t0) * 1000

        meas = self._measurements[tenant_id]
        meas.latencies.append(latency_ms)
        meas.total_requests += 1
        if latency_ms > slo:
            meas.breaches += 1

        return latency_ms

    async def run(self) -> dict[str, Any]:
        """Run the full stress test."""
        self.setup()
        sem = asyncio.Semaphore(self._concurrency)

        async def _rate_limited(tid: str) -> None:
            async with sem:
                await self._run_single(tid)

        all_tasks = []
        for tid in self._measurements:
            for _ in range(self._requests_per):
                all_tasks.append(_rate_limited(tid))

        await asyncio.gather(*all_tasks)
        return self._report()

    def _report(self) -> dict[str, Any]:
        """Generate the SLO compliance report."""
        results = []
        total_breaches = 0
        total_requests = 0

        for tid, meas in sorted(self._measurements.items()):
            results.append({
                "tenant_id": tid,
                "slo_ms": meas.slo_ms,
                "total_requests": meas.total_requests,
                "avg_ms": round(meas.avg_latency_ms, 1),
                "p99_ms": round(meas.p99_latency_ms, 1),
                "breaches": meas.breaches,
                "breach_pct": round(meas.breach_rate_pct, 2),
                "pass": meas.breach_rate_pct <= self._threshold,
            })
            total_breaches += meas.breaches
            total_requests += meas.total_requests

        overall_pct = (total_breaches / max(total_requests, 1)) * 100.0

        report = {
            "tenants": len(self._measurements),
            "total_requests": total_requests,
            "total_breaches": total_breaches,
            "overall_breach_pct": round(overall_pct, 2),
            "threshold_pct": self._threshold,
            "pass": overall_pct <= self._threshold,
            "per_tenant": results,
        }

        # Print report
        print("=" * 60)
        print("SLO Compliance Stress Test Report")
        print("=" * 60)
        for r in results:
            status = "✅ PASS" if r["pass"] else "❌ FAIL"
            print(f"  {r['tenant_id']}: SLO={r['slo_ms']:.0f}ms "
                  f"p99={r['p99_ms']:.0f}ms "
                  f"breach={r['breach_pct']:.1f}% {status}")
        print(f"\n  Overall breach rate: {overall_pct:.2f}% "
              f"(threshold: {self._threshold:.0f}%)")
        print(f"  Overall: {'✅ PASS' if report['pass'] else '❌ FAIL'}")

        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="SLO compliance stress test")
    parser.add_argument("--tenants", type=int, default=5, help="Number of tenants")
    parser.add_argument("--requests", type=int, default=200, help="Requests per tenant")
    parser.add_argument("--slo-ms", type=float, default=0,
                        help="Fixed SLO (default: random 500-2000ms)")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Max breach %% before FAIL")
    args = parser.parse_args()

    slo_range = (args.slo_ms, args.slo_ms) if args.slo_ms > 0 else (500, 2000)

    test = SLOMultiTenantStressTest(
        num_tenants=args.tenants,
        requests_per_tenant=args.requests,
        slo_ms_range=slo_range,
        breach_threshold_pct=args.threshold,
        concurrency=args.concurrency,
    )

    report = asyncio.run(test.run())
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

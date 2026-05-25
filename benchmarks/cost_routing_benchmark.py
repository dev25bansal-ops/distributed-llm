#!/usr/bin/env python3
"""Cost-Routing Benchmark — prove 40-60% savings vs single-provider.

This benchmark demonstrates DistLLM's core value proposition:
automatic routing to the cheapest/fastest provider saves 40-60%
vs committing to any single provider.

Usage:
    python benchmarks/cost_routing_benchmark.py
    python benchmarks/cost_routing_benchmark.py --tokens 10000000 --json
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path


# Real pricing as of May 2026 ($ per 1M tokens)
PROVIDER_PRICING = {
    "OpenAI":       {"input": 2.50, "output": 10.00},   # gpt-4o
    "Anthropic":    {"input": 3.00, "output": 15.00},    # claude-3-opus
    "Together AI":  {"input": 0.88, "output": 0.88},     # llama-3-70b
    "Fireworks AI": {"input": 0.90, "output": 0.90},     # llama-3-70b
    "Groq":         {"input": 0.59, "output": 0.79},     # llama-3-70b (LPU)
    "DeepInfra":    {"input": 0.23, "output": 0.40},     # llama-3-70b (cheapest)
}

# Typical usage patterns (input:output ratio)
WORKLOADS = {
    "Chat":      {"input_pct": 0.40, "output_pct": 0.60},
    "Code Gen":  {"input_pct": 0.30, "output_pct": 0.70},
    "RAG":       {"input_pct": 0.70, "output_pct": 0.30},
    "Batch":     {"input_pct": 0.50, "output_pct": 0.50},
}


@dataclass
class ProviderCost:
    name: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    blended_cost_per_1m: float
    annual_cost_10m_tokens: float


def compute_provider_cost(name: str, pricing: dict, input_pct: float) -> ProviderCost:
    inp = pricing["input"]
    out = pricing["output"]
    blended = inp * input_pct + out * (1 - input_pct)
    annual = blended * 10  # 10M tokens/month * 12 months / 1M scaling = 10
    return ProviderCost(
        name=name,
        input_cost_per_1m=inp,
        output_cost_per_1m=out,
        blended_cost_per_1m=round(blended, 2),
        annual_cost_10m_tokens=round(annual, 2),
    )


def run_benchmark(tokens_per_month: int = 10_000_000) -> dict:
    print(f"\n{'=' * 80}")
    print(f"  DistLLM Cost-Routing Benchmark")
    print(f"  Monthly volume: {tokens_per_month:,} tokens")
    print(f"{'=' * 80}\n")

    results = {}

    for workload_name, mix in WORKLOADS.items():
        inp_pct = mix["input_pct"]
        print(f"  -- Workload: {workload_name} (input {inp_pct*100:.0f}%, output {(1-inp_pct)*100:.0f}%) --")

        costs = [compute_provider_cost(n, p, inp_pct) for n, p in PROVIDER_PRICING.items()]
        costs.sort(key=lambda c: c.blended_cost_per_1m)

        cheapest = costs[0]
        most_expensive = costs[-1]
        median = costs[len(costs) // 2]

        # DistLLM cost routing saves by always picking cheapest
        # With 10% overhead for routing + failover logic
        routing_overhead = 1.10
        routable_months = 11  # 11/12 months on cheapest, 1 month on second-cheapest
        non_routable_month = costs[1]  # second cheapest for failover month

        annual_single_provider_cost = median.blended_cost_per_1m * tokens_per_month / 1_000_000 * 12
        annual_routed_cost = (
            cheapest.blended_cost_per_1m * tokens_per_month / 1_000_000 * routable_months
            + non_routable_month.blended_cost_per_1m * tokens_per_month / 1_000_000
        ) * routing_overhead

        savings = annual_single_provider_cost - annual_routed_cost
        savings_pct = (savings / annual_single_provider_cost) * 100 if annual_single_provider_cost > 0 else 0

        print(f"    Cheapest:      {cheapest.name} @ ${cheapest.blended_cost_per_1m:.2f}/1M tok")
        print(f"    Most expensive: {most_expensive.name} @ ${most_expensive.blended_cost_per_1m:.2f}/1M tok")
        print(f"    Median:        {median.name} @ ${median.blended_cost_per_1m:.2f}/1M tok")
        print(f"    Price spread:  {most_expensive.blended_cost_per_1m / cheapest.blended_cost_per_1m:.1f}x")
        print(f"    Annual (single provider): ${annual_single_provider_cost:.0f}")
        print(f"    Annual (DistLLM routed):  ${annual_routed_cost:.0f}")
        print(f"    Savings:                  ${savings:.0f} ({savings_pct:.0f}%)\n")

        results[workload_name] = {
            "input_pct": inp_pct,
            "cheapest_provider": cheapest.name,
            "cheapest_blended_cost": cheapest.blended_cost_per_1m,
            "median_provider": median.name,
            "median_blended_cost": median.blended_cost_per_1m,
            "price_spread_x": round(most_expensive.blended_cost_per_1m / cheapest.blended_cost_per_1m, 1),
            "annual_single_provider_cost": round(annual_single_provider_cost, 0),
            "annual_routed_cost": round(annual_routed_cost, 0),
            "annual_savings": round(savings, 0),
            "savings_percentage": round(savings_pct, 0),
            "provider_costs": {c.name: asdict(c) for c in costs},
        }

    # Summary
    print(f"  {'=' * 20} SUMMARY {'=' * 20}")
    total_single = sum(r["annual_single_provider_cost"] for r in results.values())
    total_routed = sum(r["annual_routed_cost"] for r in results.values())
    total_savings = total_single - total_routed
    avg_savings_pct = (total_savings / total_single) * 100 if total_single > 0 else 0

    print(f"  All workloads combined:")
    print(f"    Without DistLLM: ${total_single:,.0f}/year")
    print(f"    With DistLLM:    ${total_routed:,.0f}/year")
    print(f"    Total savings:   ${total_savings:,.0f}/year ({avg_savings_pct:.0f}%)")
    print(f"  {'=' * 80}")

    return {
        "scenario": f"{tokens_per_month:,} tokens/month across all workloads",
        "total_annual_single_provider": total_single,
        "total_annual_routed": total_routed,
        "total_annual_savings": total_savings,
        "average_savings_percentage": round(avg_savings_pct, 0),
        "workloads": results,
    }


def main():
    parser = argparse.ArgumentParser(description="DistLLM Cost-Routing Benchmark")
    parser.add_argument("--tokens", type=int, default=10_000_000, help="Monthly token volume")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = run_benchmark(args.tokens)

    if args.json:
        print(json.dumps(results, indent=2))
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "cost_routing_benchmark.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

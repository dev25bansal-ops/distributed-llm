"""``distllm cost-avoid`` — calculate savings from self-hosted inference.

Estimates the monthly cost of running a model on cloud APIs vs. self-hosting
on your own hardware with DistLLM.

Usage::

    distllm cost-avoid --model meta-llama/Llama-3.1-70B --requests-per-day 50000
"""

import argparse

# Cloud GPU pricing per hour (USD) — updated periodically
_CLOUD_GPU_PRICING = {
    "A100-80GB": 3.50,
    "A100-40GB": 1.50,
    "H100": 4.50,
    "RTX 4090": 0.50,
    "RTX 3090": 0.35,
    "RTX 3080": 0.25,
    "RTX 4060": 0.15,
}

# API provider pricing per million input tokens (USD)
_API_INPUT_PRICING = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "claude-3-haiku": 0.25,
    "claude-3-sonnet": 3.00,
    "llama-3.1-70b-together": 0.90,
    "llama-3.1-70b-deepinfra": 0.40,
    "llama-3.1-8b-together": 0.18,
}

# Model parameter counts for VRAM estimation
_MODEL_VRAM_GB = {
    "70b": 140,
    "40b": 80,
    "13b": 26,
    "8b": 16,
    "7b": 14,
    "3b": 6,
    "1b": 2,
}


def _estimate_model_size(model_name: str) -> int:
    """Estimate model size in billions of parameters from model name."""
    name_lower = model_name.lower()
    for size_str in ["70b", "40b", "13b", "8b", "7b", "3b", "1b"]:
        if size_str in name_lower:
            return int(size_str.replace("b", ""))
    return 7  # Default to 7B


def _estimate_vram_gb(model_size_b: int, quant: str = "fp16") -> int:
    """Estimate VRAM needed for a model."""
    base = _MODEL_VRAM_GB.get(f"{model_size_b}b", model_size_b * 2)
    if quant == "int8":
        return base // 2
    if quant == "int4":
        return base // 4
    return base


def calculate_cost_avoidance(
    model_name: str,
    requests_per_day: int = 10000,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 1000,
    gpu_type: str = "RTX 4090",
    cloud_api: str = "llama-3.1-70b-deepinfra",
    days_per_month: int = 30,
) -> dict:
    """Calculate monthly savings from self-hosting.

    Args:
        model_name: HuggingFace model name.
        requests_per_day: Average daily request volume.
        avg_input_tokens: Average prompt length in tokens.
        avg_output_tokens: Average generation length in tokens.
        gpu_type: GPU model for self-hosting cost.
        cloud_api: Cloud API for comparison pricing.
        days_per_month: Billing days per month.

    Returns:
        Dict with cost comparison breakdown.
    """
    model_size_b = _estimate_model_size(model_name)
    vram_gb = _estimate_vram_gb(model_size_b)

    # GPU count needed
    gpu_hourly = _CLOUD_GPU_PRICING.get(gpu_type, 1.00)
    gpus_needed = max(1, vram_gb // 48 + 1)  # Assume ~48GB usable per GPU

    # Self-hosting cost
    monthly_gpu_cost = gpu_hourly * gpus_needed * 24 * days_per_month
    # Assume ~20% overhead for power, cooling, networking
    monthly_total_self = monthly_gpu_cost * 1.2

    # Cloud API cost
    api_price_per_m = _API_INPUT_PRICING.get(cloud_api, 0.50)
    monthly_input_tokens = requests_per_day * avg_input_tokens * days_per_month
    monthly_output_tokens = requests_per_day * avg_output_tokens * days_per_month
    monthly_api_cost = (
        monthly_input_tokens / 1_000_000 * api_price_per_m
        + monthly_output_tokens / 1_000_000 * api_price_per_m * 3  # Output is ~3x more expensive
    )

    monthly_savings = max(0, monthly_api_cost - monthly_total_self)
    payback_days = (gpus_needed * 3000) / max(monthly_savings / days_per_month, 1) if monthly_savings > 0 else float("inf")

    return {
        "model": model_name,
        "model_size_b": model_size_b,
        "estimated_vram_gb": vram_gb,
        "gpus_needed": gpus_needed,
        "requests_per_day": requests_per_day,
        "comparison_api": cloud_api,
        "monthly_api_cost": round(monthly_api_cost, 2),
        "monthly_self_hosted_cost": round(monthly_total_self, 2),
        "monthly_savings": round(monthly_savings, 2),
        "savings_percent": round(monthly_savings / max(monthly_api_cost, 1) * 100, 1),
        "payback_period_days": round(payback_days, 1) if payback_days < 3650 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate cloud cost avoidance from self-hosted inference")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B", help="Model name")
    parser.add_argument("--requests-per-day", type=int, default=10000, help="Daily request volume")
    parser.add_argument("--avg-input-tokens", type=int, default=500, help="Average prompt length")
    parser.add_argument("--avg-output-tokens", type=int, default=1000, help="Average generation length")
    parser.add_argument("--gpu-type", default="RTX 4090", choices=list(_CLOUD_GPU_PRICING.keys()), help="Your GPU type")
    parser.add_argument("--cloud-api", default="llama-3.1-70b-deepinfra", help="Cloud API for comparison")
    args = parser.parse_args()

    result = calculate_cost_avoidance(
        model_name=args.model,
        requests_per_day=args.requests_per_day,
        avg_input_tokens=args.avg_input_tokens,
        avg_output_tokens=args.avg_output_tokens,
        gpu_type=args.gpu_type,
        cloud_api=args.cloud_api,
    )

    print(f"\n{'='*55}")
    print(f"  Cost Avoidance Calculator")
    print(f"{'='*55}")
    print(f"  Model:              {result['model']}")
    print(f"  Size:               {result['model_size_b']}B parameters")
    print(f"  VRAM needed:        ~{result['estimated_vram_gb']} GB")
    print(f"  GPUs needed:        {result['gpus_needed']}x {args.gpu_type}")
    print(f"  Requests/day:       {result['requests_per_day']:,}")
    print(f"{'='*55}")
    print(f"  Cloud API cost:     ${result['monthly_api_cost']:,.2f}/mo")
    print(f"  Self-hosted cost:   ${result['monthly_self_hosted_cost']:,.2f}/mo")
    print(f"  Monthly savings:    ${result['monthly_savings']:,.2f}/mo")
    print(f"  Savings:            {result['savings_percent']}%")
    if result['payback_period_days'] < 3650:
        print(f"  GPU payback:        ~{result['payback_period_days']} days")
    else:
        print(f"  GPU payback:        N/A (no savings)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

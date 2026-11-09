"""``distllm tutorial`` — Interactive guided setup for first-time users.

Walks through:
  1. Checking system requirements (CUDA, Python, disk)
  2. Installing DistLLM
  3. Starting a coordinator
  4. Running inference
  5. Adding worker nodes
"""

import os
import sys
import time
from typing import NoReturn

import torch


def _print_step(num: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {num}: {title}")
    print(f"{'='*60}")


def _wait(seconds: float = 2.0) -> None:
    time.sleep(seconds)


def _check_mark() -> str:
    return "✅"


def _run_tutorial() -> None:
    print()
    print("  Welcome to DistLLM — Distributed Inference Across All Your Devices!")
    print("  This tutorial will guide you through setting up your first cluster.")
    print()

    _print_step(1, "System Requirements Check")
    print()
    try:
        version = getattr(torch, "__version__", "unknown")
        print(f"  {_check_mark()} PyTorch {version}")
        cuda = bool(torch.cuda.is_available())
        if cuda:
            count = int(torch.cuda.device_count())
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                print(f"  {_check_mark()} GPU {i}: {name}")
        else:
            print("  ⚠️  No CUDA GPU detected. DistLLM will use CPU (slow).")
    except Exception:
        # CQ-08: failure is surfaced via console output; not propagated.
        print("  ❌ PyTorch not available. Run: pip install distllm[self-hosted]")
        return

    print()
    input("  Press Enter to continue...")

    _print_step(2, "Installation Check")
    print()
    try:
        import distllm
        print(f"  {_check_mark()} DistLLM {distllm.__version__ if hasattr(distllm, '__version__') else 'installed'}")
    except ImportError:
        print("  ❌ DistLLM not installed. Run: pip install distllm")
        return

    print()
    input("  Press Enter to continue...")

    _print_step(3, "Starting a Local Coordinator")
    print()
    print("  Run this command in a terminal:")
    print()
    print("    distllm cluster start --model HuggingFaceTB/SmolLM-135M --local")
    print()
    print("  This downloads a small 135M parameter model and starts")
    print("  an OpenAI-compatible API server on http://localhost:8000")
    print()
    print("  Note: First download may take 1-2 minutes.")

    print()
    input("  Press Enter after starting the coordinator...")

    _print_step(4, "Testing Inference")
    print()
    print("  In another terminal, run:")
    print()
    print("    distllm chat")
    print()
    print("  Or using curl:")
    print()
    print('    curl http://localhost:8000/v1/chat/completions \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -H "Authorization: Bearer $API_KEY" \\')
    print('      -d \'{"messages":[{"role":"user","content":"Hello!"}],"model":"default"}\'')
    print()

    _print_step(5, "Adding Worker Nodes (Multi-Device)")
    print()
    print("  On each additional machine, run:")
    print()
    print("    distllm cluster join --coordinator <coordinator-ip>:50050")
    print()
    print("  The coordinator automatically assigns layers based on")
    print("  each device's available GPU memory.")
    print()

    _print_step(6, "Monitor Your Cluster")
    print()
    print("  Open http://localhost:8000/dashboard in your browser")
    print("  to see real-time metrics: GPU usage, latency, throughput.")
    print()

    print(f"{'='*60}")
    print("  You're ready to run distributed inference!")
    print("  Next steps:")
    print("    - Try larger models: distllm cluster start --model meta-llama/Llama-3.2-7B")
    print("    - Run benchmarks: distllm benchmark --model HuggingFaceTB/SmolLM-135M")
    print("    - Check docs: https://github.com/distributed-llm/distributed-llm#readme")
    print(f"{'='*60}")
    print()


def main() -> None:
    try:
        _run_tutorial()
    except KeyboardInterrupt:
        print("\n\n  Tutorial interrupted. Goodbye!")
        sys.exit(0)

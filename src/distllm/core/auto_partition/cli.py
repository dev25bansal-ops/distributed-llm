"""CLI entry point for running the hardware-aware partitioner standalone."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from loguru import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hardware-Aware Auto-Partitioner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Quick run with default 7B model on detected GPUs:\n"
            "  python -m distllm.core.auto_partition.cli\n\n"
            "  # 70B model across 4 heterogeneous nodes:\n"
            "  python -m distllm.core.auto_partition.cli \\\n"
            "    --hidden-size 8192 --intermediate-size 28672 \\\n"
            "    --num-layers 80 --nodes node-a,node-b,node-c,node-d \\\n"
            "    --gpu-counts node-a:1,node-b:2,node-c:1,node-d:2 \\\n"
            "    --batch-size 4 --seq-len 8192 --compare\n\n"
            "  # Save and load plans:\n"
            "  python -m distllm.core.auto_partition.cli --save plan.json\n"
            "  python -m distllm.core.auto_partition.cli --load plan.json\n"
        ),
    )

    # Model architecture
    parser.add_argument("--hidden-size", type=int, default=4096, help="Model hidden dimension")
    parser.add_argument("--intermediate-size", type=int, default=11008, help="MLP intermediate size")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")

    # Nodes / hardware
    parser.add_argument("--nodes", type=str, default=None, help="Comma-separated node IDs")
    parser.add_argument("--gpu-counts", type=str, default=None, help="Node:gpu_count pairs (e.g., node-a:1,node-b:2)")
    parser.add_argument("--hostnames", type=str, default=None, help="Node:hostname pairs (e.g., node-a:10.0.0.1,node-b:10.0.0.2)")

    # Workload
    parser.add_argument("--batch-size", type=int, default=1, help="Target batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Target sequence length")

    # Comparison
    parser.add_argument("--compare", action="store_true", help="Compare DP vs equal/proportional splits")
    parser.add_argument("--save", type=str, default=None, help="Save partition plan to JSON file")
    parser.add_argument("--load", type=str, default=None, help="Load and display a saved plan")

    # Verbosity
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser


async def run_partitioner(args: argparse.Namespace) -> None:
    """Run the partitioner with CLI arguments and display results."""
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Parse nodes
    node_ids: list[str] | None = None
    if args.nodes:
        node_ids = [n.strip() for n in args.nodes.split(",")]

    # Parse GPU counts
    gpu_counts: dict[str, int] | None = None
    if args.gpu_counts:
        gpu_counts = {}
        for pair in args.gpu_counts.split(","):
            node, count = pair.strip().split(":")
            gpu_counts[node] = int(count)

    # Parse hostnames
    hostnames: dict[str, str] | None = None
    if args.hostnames:
        hostnames = {}
        for pair in args.hostnames.split(","):
            node, host = pair.strip().split(":")
            hostnames[node] = host

    from distllm.core.auto_partition.partitioner import HardwareAwarePartitioner

    partitioner = HardwareAwarePartitioner(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    solution = await partitioner.partition(
        model_name="cli_model",
        node_ids=node_ids,
        gpu_counts=gpu_counts,
        hostnames=hostnames,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
    )

    print()
    print(solution.summary())
    print()

    # Display per-node summaries
    summaries = partitioner.get_node_summaries()
    if summaries:
        print("Per-node breakdown:")
        for s in summaries:
            mem_str = f"{s['memory_gb']:.1f}GB" if s['fits_in_memory'] else f"{s['memory_gb']:.1f}GB [OOM!]"
            print(
                f"  {s['node_id']}: layers {s['layers']} "
                f"({s['num_layers']} layers) "
                f"compute={s['compute_time_ms']:.1f}ms "
                f"comm={s['comm_time_ms']:.1f}ms "
                f"total={s['total_time_ms']:.1f}ms "
                f"mem={mem_str}"
            )
    print()

    # Optional comparison
    if args.compare:
        print("Strategy comparison:")
        comparison = partitioner.compare_to_baselines()
        if comparison:
            for strategy, metrics in comparison.items():
                if isinstance(metrics, dict):
                    print(
                        f"  {strategy}: "
                        f"max_latency={metrics.get('max_latency_ms', 'N/A')}ms, "
                        f"throughput={metrics.get('throughput', 'N/A')} tok/s"
                    )
                else:
                    print(f"  {strategy}: {metrics}")
        print()

    # Optional save
    if args.save:
        with open(args.save, "w") as f:
            data = {
                "solution": {
                    "max_node_time_ms": solution.max_node_time_ms,
                    "estimated_throughput_tok_s": solution.estimated_throughput_tok_s,
                    "assignments": [
                        {
                            "node_id": p.node_id,
                            "start_layer": p.start_layer,
                            "end_layer": p.end_layer,
                            "estimated_time_ms": p.estimated_time_ms,
                        }
                        for p in solution.points
                    ],
                },
                "config": {
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                },
            }
            json.dump(data, f, indent=2)
        print(f"Plan saved to {args.save}")
        print()


def display_plan(path: str) -> None:
    """Load and display a saved partition plan."""
    import json

    with open(path) as f:
        data = json.load(f)

    print("Saved Partition Plan:")
    print(f"  Max node time: {data['solution']['max_node_time_ms']}ms")
    print(f"  Est. throughput: {data['solution']['estimated_throughput_tok_s']} tok/s")
    print("  Assignments:")
    for a in data["solution"]["assignments"]:
        print(f"    {a['node_id']}: layers [{a['start_layer']}, {a['end_layer']}) ~{a['estimated_time_ms']}ms")
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.load:
        display_plan(args.load)
        return

    asyncio.run(run_partitioner(args))


if __name__ == "__main__":
    main()

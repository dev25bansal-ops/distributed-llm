"""Command-line interface for the DistLLM SDK.

Usage::

    distllm chat --model <model> --message <msg> [--stream] [--temperature 0.7]
    distllm complete --model <model> --prompt <text> [--max-tokens 256]
    distllm embed --input <text> [--model <model>]
    distllm models
    distllm health
    distllm benchmark --model <model> --prompts <file> [--num-runs 5]
    distllm eval --model <model> --questions <file> --answers <file>
                 [--metrics relevancy,faithfulness]

Global options::

    --base-url  http://localhost:8000
    --api-key   (default from DISTLLM_API_KEY env var)
    --timeout   120
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colour helpers (no external dependencies)
# ---------------------------------------------------------------------------

_COLORS = {
    "green": "\x1b[32m",
    "cyan": "\x1b[36m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "reset": "\x1b[0m",
}


def _c(code: str, text: str) -> str:
    """Wrap *text* with the ANSI escape for *code*."""
    return f"{_COLORS[code]}{text}{_COLORS['reset']}"


def _bold(text: str) -> str:
    return _c("bold", text)


def _green(text: str) -> str:
    return _c("green", text)


def _yellow(text: str) -> str:
    return _c("yellow", text)


def _red(text: str) -> str:
    return _c("red", text)


def _dim(text: str) -> str:
    return _c("dim", text)


def _stderr(*args: object) -> None:
    """Print to stderr."""
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the *p*-th percentile (0-100) of an already-sorted list."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _resolve_api_key(api_key: str | None) -> str | None:
    """Return *api_key* if given, otherwise fall back to the environment."""
    if api_key:
        return api_key
    return os.environ.get("DISTLLM_API_KEY")


def _read_prompts(path_str: str) -> list[str]:
    """Read prompts from *path_str*, one prompt per non-empty line."""
    path = Path(path_str)
    if not path.exists():
        _stderr(_red(f"ERROR: prompts file not found: {path}"))
        sys.exit(1)
    prompts = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        _stderr(_red("ERROR: prompts file is empty"))
        sys.exit(1)
    return prompts


def _read_questions(path_str: str, label: str) -> list[str]:
    """Read textual content from *path_str*, one item per line.

    If the file has a ``.json`` or ``.jsonl`` extension, attempt to read it as
    structured data instead.
    """
    path = Path(path_str)
    if not path.exists():
        _stderr(_red(f"ERROR: {label} file not found: {path}"))
        sys.exit(1)

    if path.suffix in (".json", ".jsonl"):
        items: list[str] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    _stderr(_yellow(f"WARNING: skipping unparseable JSON line in {label}"))
                    continue
                if isinstance(obj, str):
                    items.append(obj)
                elif isinstance(obj, dict):
                    items.append(obj.get("question") or obj.get("text") or json.dumps(obj))
                else:
                    items.append(str(obj))
        if not items:
            _stderr(_red(f"ERROR: {label} file contains no usable content"))
            sys.exit(1)
        return items

    items = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        _stderr(_red(f"ERROR: {label} file is empty"))
        sys.exit(1)
    return items


# ---------------------------------------------------------------------------
# Subcommand: chat
# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> None:
    """Send a chat completion message."""
    # Lazy import so ``distllm --help`` doesn't force the SDK import.
    from distllm_sdk.client import DistLLMClientSync  # type: ignore[import-untyped] # noqa: PLC0415

    client = DistLLMClientSync(
        base_url=args.base_url,
        api_key=_resolve_api_key(args.api_key),
        timeout=args.timeout,
    )

    messages = [{"role": "user", "content": args.message}]

    if args.stream:
        for chunk in client.chat_completions_stream(
            messages=messages,
            model=args.model,
            temperature=args.temperature,
        ):
            print(chunk, end="", flush=True)
        print()
        return

    resp = client.chat_completions(
        messages=messages,
        model=args.model,
        temperature=args.temperature,
    )

    if resp.choices:
        msg = resp.choices[0].message
        print(msg.content if msg else "")
    else:
        _stderr(_yellow("No response choices returned."))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: complete
# ---------------------------------------------------------------------------

def cmd_complete(args: argparse.Namespace) -> None:
    """Send a text completion request."""
    from distllm_sdk.client import DistLLMClientSync  # type: ignore[import-untyped] # noqa: PLC0415

    client = DistLLMClientSync(
        base_url=args.base_url,
        api_key=_resolve_api_key(args.api_key),
        timeout=args.timeout,
    )

    resp = client.completions(
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if resp.choices:
        for choice in resp.choices:
            print(choice.text)
    else:
        _stderr(_yellow("No completion choices returned."))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: embed
# ---------------------------------------------------------------------------

def cmd_embed(args: argparse.Namespace) -> None:
    """Generate embeddings."""
    from distllm_sdk.client import DistLLMClientSync  # type: ignore[import-untyped] # noqa: PLC0415

    client = DistLLMClientSync(
        base_url=args.base_url,
        api_key=_resolve_api_key(args.api_key),
        timeout=args.timeout,
    )

    resp = client.embeddings(input=args.input, model=args.model)

    for idx, obj in enumerate(resp.data):
        vec = obj.embedding
        n_dims = len(vec) if isinstance(vec, list) else 0
        preview = _dim(f"[{n_dims} floats]")
        index_str = _bold(f"[{idx}]")
        print(f"{index_str} embedding {preview}")
        if args.verbose:
            # Print the full vector on a second line.
            print(f"       {vec}")


# ---------------------------------------------------------------------------
# Subcommand: models
# ---------------------------------------------------------------------------

def cmd_models(args: argparse.Namespace) -> None:
    """List available models."""
    from distllm_sdk.client import DistLLMClientSync  # type: ignore[import-untyped] # noqa: PLC0415

    client = DistLLMClientSync(
        base_url=args.base_url,
        api_key=_resolve_api_key(args.api_key),
        timeout=args.timeout,
    )

    models = client.list_models()

    header = _bold(f"{'Model ID':<50} {'Owned By'}")
    print(header)
    print(_dim("-" * 80))
    for m in models.data:
        print(f"{m.id:<50} {m.owned_by}")

    print(_dim(f"\nTotal: {len(models.data)} model(s)"))


# ---------------------------------------------------------------------------
# Subcommand: health
# ---------------------------------------------------------------------------

def cmd_health(args: argparse.Namespace) -> None:
    """Check cluster health."""
    from distllm_sdk.client import DistLLMClientSync  # type: ignore[import-untyped] # noqa: PLC0415

    client = DistLLMClientSync(
        base_url=args.base_url,
        api_key=_resolve_api_key(args.api_key),
        timeout=args.timeout,
    )

    try:
        data = client.health_check()
    except Exception as exc:
        _stderr(_red(f"Health check FAILED: {exc}"))
        sys.exit(1)

    status = data.get("status", data.get("health", "unknown"))
    if status in ("ok", "healthy", "up"):
        print(_green(f"Cluster status: {status}"))
    else:
        print(_yellow(f"Cluster status: {status}"))

    # Pretty-print any extra fields.
    for key, val in data.items():
        if key in ("status", "health"):
            continue
        label = _bold(f"  {key}:")
        if isinstance(val, dict) or isinstance(val, list):
            print(label)
            print(_dim(f"    {json.dumps(val, indent=4)}".replace("\n", "\n    ")))
        else:
            print(f"{label} {val}")


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run a performance benchmark against a model."""
    from distllm_sdk.client import DistLLMClientSync  # type: ignore[import-untyped] # noqa: PLC0415

    prompts = _read_prompts(args.prompts)
    n_runs = args.num_runs
    model = args.model

    # Pre-warm: build the client once.
    client = DistLLMClientSync(
        base_url=args.base_url,
        api_key=_resolve_api_key(args.api_key),
        timeout=args.timeout,
    )

    latencies: list[float] = []
    tokens_per_sec: list[float] = []
    total_tokens: int = 0
    errors: int = 0

    print(_bold(f"Benchmark: model={model}  runs={n_runs}  prompts={len(prompts)}"))
    print(_dim("-" * 60))

    for i in range(n_runs):
        prompt_text = prompts[i % len(prompts)]
        start = time.monotonic()
        try:
            resp = client.chat_completions(
                messages=[{"role": "user", "content": prompt_text}],
                model=model,
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as exc:
            _stderr(_red(f"  Run {i + 1}/{n_runs} ERROR: {exc}"))
            errors += 1
            continue

        elapsed = time.monotonic() - start
        latencies.append(elapsed)

        if resp.usage:
            tps = resp.usage.tokens_per_second or 0.0
            if tps > 0:
                tokens_per_sec.append(tps)
            total_tokens += resp.usage.completion_tokens

        bar = _green(".") if elapsed < 5 else _yellow(".")
        print(f"  {bar} Run {i + 1:>{len(str(n_runs))}}/{n_runs}  "
              f"{elapsed:.2f}s  "
              f"{'(error)' if resp.choices is None or len(resp.choices) == 0 else ''}",
              end="",
              flush=True)

    print(_dim("\n" + "-" * 60))
    print()

    # --- Summary -----------------------------------------------------------
    if latencies:
        sorted_lat = sorted(latencies)
        avg = statistics.mean(latencies)
        p50 = _percentile(sorted_lat, 50)
        p95 = _percentile(sorted_lat, 95)
        p99 = _percentile(sorted_lat, 99)
        _min = sorted_lat[0]
        _max = sorted_lat[-1]

        avg_tps = statistics.mean(tokens_per_sec) if tokens_per_sec else 0.0

        print(_bold("Latency (seconds):"))
        print(f"  Average:  {avg:.2f}s")
        print(f"  P50:      {p50:.2f}s")
        print(f"  P95:      {p95:.2f}s")
        print(f"  P99:      {p99:.2f}s")
        print(f"  Min:      {_min:.2f}s")
        print(f"  Max:      {_max:.2f}s")

        print(_bold("\nThroughput:"))
        print(f"  Tokens/sec (avg): {avg_tps:.1f}")
        print(f"  Total tokens:     {total_tokens}")
        print(f"  Success rate:     {(n_runs - errors)}/{n_runs}")

        # Return exit code 0 if at least half succeeded.
        if errors > n_runs // 2:
            _stderr(_red("ERROR: more than half of benchmark runs failed."))
            sys.exit(1)
    else:
        _stderr(_red("No successful benchmark runs completed."))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------

_EVAL_METRICS = ("relevancy", "faithfulness", "accuracy", "precision", "recall", "f1")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run a quick evaluation using EvalHarness."""
    questions = _read_questions(args.questions, "questions")
    answers = _read_questions(args.answers, "answers")
    metrics_str = args.metrics or "relevancy,faithfulness"
    requested_metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]

    for m in requested_metrics:
        if m not in _EVAL_METRICS:
            _stderr(_yellow(f"WARNING: unknown metric '{m}' (known: {', '.join(_EVAL_METRICS)})"))

    if len(questions) != len(answers):
        _stderr(_red(
            f"ERROR: question count ({len(questions)}) does not match "
            f"answer count ({len(answers)})"
        ))
        sys.exit(1)

    # Lazy-import EvalHarness (may not be installed).
    try:
        from eval_harness import EvalHarness  # type: ignore[import-untyped] # noqa: PLC0415
    except ImportError:
        _stderr(_red(
            "ERROR: 'eval_harness' package is not installed.\n"
            "  Install with: pip install distllm-sdk[eval]  "
            "(or pip install eval-harness)"
        ))
        sys.exit(1)

    # Optional: wire the model through DistLLM if EvalHarness supports it.
    try:
        harness = EvalHarness(model=args.model)
    except Exception as exc:
        _stderr(_red(f"ERROR: failed to initialise EvalHarness: {exc}"))
        sys.exit(1)

    print(_bold(f"Evaluating {len(questions)} Q&A pairs..."))
    print(_dim(f"  Model:     {args.model}"))
    print(_dim(f"  Metrics:   {', '.join(requested_metrics)}"))
    print(_dim("-" * 60))

    results: dict[str, float] = {}
    try:
        results = harness.evaluate(
            questions=questions,
            answers=answers,
            metrics=requested_metrics,
        )
    except Exception as exc:
        _stderr(_red(f"ERROR: evaluation failed: {exc}"))
        sys.exit(1)

    print()
    print(_bold("Results:"))
    for metric_name in requested_metrics:
        score = results.get(metric_name, None)
        if score is None:
            print(f"  {metric_name}: {_yellow('N/A')}")
        else:
            formatted = _green(f"{score:.3f}") if score >= 0.5 else _yellow(f"{score:.3f}")
            print(f"  {metric_name}: {formatted}")

    # Optionally dump full results as JSON.
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps({
                "model": args.model,
                "metrics": requested_metrics,
                "results": results,
                "count": len(questions),
            }, indent=2),
            encoding="utf-8",
        )
        print(_dim(f"\nFull results written to: {output_path}"))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distllm",
        description=_bold("Command-line interface for the DistLLM SDK."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "See https://github.com/distributed-llm for full documentation.\n"
            "\n"
            "Environment variables:\n"
            "  DISTLLM_API_KEY   API key (alternative to --api-key)\n"
        ),
    )

    # ----- Global options --------------------------------------------------
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Cluster base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: DISTLLM_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )

    # ----- Subcommands -----------------------------------------------------
    sub = parser.add_subparsers(dest="command", required=True, title="Commands")

    # chat
    p_chat = sub.add_parser("chat", help="Send a chat completion message")
    p_chat.add_argument("--model", default="distributed-llm", help="Model identifier")
    p_chat.add_argument("--message", "-m", required=True, help="User message text")
    p_chat.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    p_chat.add_argument("--stream", action="store_true", help="Stream response tokens")
    p_chat.set_defaults(func=cmd_chat)

    # complete
    p_comp = sub.add_parser("complete", help="Send a text completion request")
    p_comp.add_argument("--model", default="distributed-llm", help="Model identifier")
    p_comp.add_argument("--prompt", required=True, help="Prompt text")
    p_comp.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate (default: 256)")
    p_comp.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    p_comp.set_defaults(func=cmd_complete)

    # embed
    p_embed = sub.add_parser("embed", help="Generate embeddings for input text")
    p_embed.add_argument("--input", required=True, help="Text to embed")
    p_embed.add_argument("--model", default="distributed-llm", help="Model identifier")
    p_embed.add_argument("--verbose", "-v", action="store_true", help="Print full embedding vector")
    p_embed.set_defaults(func=cmd_embed)

    # models
    p_models = sub.add_parser("models", help="List available models")
    p_models.set_defaults(func=cmd_models)

    # health
    p_health = sub.add_parser("health", help="Check cluster health")
    p_health.set_defaults(func=cmd_health)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run a performance benchmark")
    p_bench.add_argument("--model", default="distributed-llm", help="Model identifier")
    p_bench.add_argument("--prompts", required=True, help="Path to prompts file (one per line)")
    p_bench.add_argument("--num-runs", type=int, default=5, help="Number of benchmark runs (default: 5)")
    p_bench.set_defaults(func=cmd_benchmark)

    # eval
    p_eval = sub.add_parser("eval", help="Run a quick evaluation")
    p_eval.add_argument("--model", default="distributed-llm", help="Model identifier")
    p_eval.add_argument("--questions", required=True, help="Path to questions file")
    p_eval.add_argument("--answers", required=True, help="Path to answers file")
    p_eval.add_argument(
        "--metrics",
        default="relevancy,faithfulness",
        help="Comma-separated metric names (default: relevancy,faithfulness)",
    )
    p_eval.add_argument(
        "--output", "-o",
        default=None,
        help="Write full evaluation results as JSON to this path",
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point.  Parses arguments and dispatches to the appropriate
    subcommand function.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except KeyboardInterrupt:
        _stderr(_yellow("\nInterrupted."))
        sys.exit(130)
    except Exception as exc:
        _stderr(_red(f"ERROR: {exc}"))
        if os.environ.get("DISTLLM_CLI_DEBUG"):
            import traceback  # noqa: PLC0415
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

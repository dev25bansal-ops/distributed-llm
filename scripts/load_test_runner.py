"""In-process load-test harness: starts the DistLLM API server with a
deterministic mock inference backend, runs the locustfile against it, and
prints requests/sec + latency percentiles.

This measures the full HTTP stack (middleware pipeline, auth, rate limiting,
serialization) with generation stubbed out — useful for capacity analysis of
the serving layer itself. It is NOT a model-throughput benchmark.

Usage::

    python scripts/load_test_runner.py [--users 10] [--runtime 60s] [--port 8902]
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import platform
import subprocess
import sys
import threading
import time

# ── Environment must be set BEFORE importing distllm.api.server ──────────────
# (middleware reads these at class-definition/import time)
TEST_API_KEY = "loadtest-" + os.urandom(16).hex()
os.environ["API_KEY"] = TEST_API_KEY
# Effectively disable per-IP request rate limiting for the run.
os.environ.setdefault("DISTLLM_RATE_LIMIT_REQUESTS", "10000000")


def build_mock_coordinator():
    """Deterministic zero-computation coordinator mirroring tests' mocks."""
    from unittest.mock import MagicMock

    import torch

    from distllm.api.api_state import g

    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None          # BackpressureMiddleware skips when falsy
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._model_router = None
    coord._shutting_down = False

    def encode_fn(text, **kwargs):
        tokens = [1, 2, 3, 4, 5]
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.side_effect = encode_fn
    # TemplateEngine falls through to naive join when no chat template.
    coord.tokenizer.chat_template = None

    coord.generate.return_value = "Deterministic load-test response."
    coord.list_models.return_value = ["test-model"]
    g.coordinator = coord
    return coord


def start_server(port: int):
    """Run uvicorn (full app + lifespan/plugins) in a daemon thread."""
    import uvicorn

    from distllm.api.server import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import urllib.request

    deadline = time.time() + 90
    base = f"http://127.0.0.1:{port}"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                if r.status == 200:
                    return server, base
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Server did not become healthy within 90s")


def run_locust(locustfile: str, host: str, users: int, runtime: str, out_prefix: str) -> None:
    cmd = [
        sys.executable, "-m", "locust",
        "-f", locustfile,
        "--headless",
        "-u", str(users),
        "-r", "2",
        "--run-time", runtime,
        "--host", host,
        "--csv", out_prefix,
        "--only-summary",
    ]
    subprocess.run(cmd, check=True)


def summarize(csv_prefix: str, users: int, runtime_s: float) -> dict:
    path = f"{csv_prefix}_stats.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    agg = next(r for r in rows if r["Name"] == "Aggregated")
    return {
        "users": users,
        "runtime_s": runtime_s,
        "requests": int(agg["Request Count"]),
        "failures": int(agg["Failure Count"]),
        "rps": float(agg["Requests/s"]),
        "median_ms": float(agg["Median Response Time"]),
        "avg_ms": float(agg["Average Response Time"]),
        "p95_ms": float(agg["95%"]),
        "p99_ms": float(agg["99%"]),
        "max_ms": float(agg["Max Response Time"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--runtime", type=str, default="60s")
    parser.add_argument("--port", type=int, default=8902)
    args = parser.parse_args()

    runtime_s = float(args.runtime.rstrip("sS") or 60)

    print(f"[loadtest] python {sys.version.split()[0]} on {platform.system()} "
          f"{platform.release()} ({multiprocessing.cpu_count()} logical CPUs)")
    print(f"[loadtest] building deterministic mock coordinator...")
    build_mock_coordinator()
    print(f"[loadtest] starting server on 127.0.0.1:{args.port} ...")
    server, base = start_server(args.port)
    print(f"[loadtest] server healthy at {base}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    locustfile = os.path.join(script_dir, "locustfile.py")
    out_prefix = os.path.abspath("_loadtest_results")

    env = os.environ.copy()
    env["API_KEY"] = TEST_API_KEY
    os.environ["API_KEY"] = TEST_API_KEY  # locustfile reads this in child too

    print(f"[loadtest] running locust: {args.users} users for {args.runtime} ...")
    try:
        run_locust(locustfile, base, args.users, args.runtime, out_prefix)
    finally:
        server.should_exit = True
        time.sleep(1)

    s = summarize(out_prefix, args.users, runtime_s)
    fail_pct = 100.0 * s["failures"] / s["requests"] if s["requests"] else 0.0
    print("\n=== RESULTS ===")
    print(f"users:            {s['users']}")
    print(f"duration_s:       {s['runtime_s']:.0f}")
    print(f"total_requests:   {s['requests']}")
    print(f"failures:         {s['failures']} ({fail_pct:.2f}%)")
    print(f"throughput_rps:   {s['rps']:.1f}")
    print(f"latency_median_ms: {s['median_ms']}")
    print(f"latency_avg_ms:   {s['avg_ms']:.1f}")
    print(f"latency_p95_ms:   {s['p95_ms']}")
    print(f"latency_p99_ms:   {s['p99_ms']}")
    print(f"latency_max_ms:   {s['max_ms']}")


if __name__ == "__main__":
    main()

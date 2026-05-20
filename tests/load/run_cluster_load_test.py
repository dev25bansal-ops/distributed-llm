"""CI load-test orchestrator: starts coordinator + N nodes, runs Locust, checks SLOs.

Usage:
    python tests/load/run_cluster_load_test.py [--model MODEL] [--api-port PORT]
        [--users 50] [--run-time 3m] [--p95-ms 5000] [--error-rate 0.01]
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
LOCUST_SCENARIO = HERE / "locust" / "locustfile.py"

MODEL = "roneneldan/TinyStories-1M"
# 8 layers total => node_0 gets 0-3, node_1 gets 4-7
TOTAL_LAYERS = 8

FAILED = False


def log(msg: str) -> None:
    print(f"[load-test] {msg}", flush=True)


def check_slo(csv_path: str, p95_ms: float, error_rate: float) -> bool:
    """Returns True if all SLOs are met."""
    if not os.path.exists(csv_path):
        log(f"SLO CSV not found: {csv_path}")
        return False

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") == "Aggregated":
                p95_actual = float(row.get("95% Response Time", 0))
                total = int(row.get("Request Count", 0))
                failures = int(row.get("Failure Count", 0))
                actual_error_rate = failures / max(total, 1)

                log(f"  p95 latency: {p95_actual:.0f}ms (threshold: {p95_ms:.0f}ms)")
                log(f"  error rate:  {actual_error_rate:.4f} (threshold: {error_rate:.4f})")
                log(f"  requests:    {total}")

                ok = True
                if p95_actual > p95_ms:
                    log(f"  ✗ p95 {p95_actual:.0f}ms > {p95_ms:.0f}ms")
                    ok = False
                if actual_error_rate > error_rate:
                    log(f"  ✗ error rate {actual_error_rate:.4f} > {error_rate:.4f}")
                    ok = False
                if ok:
                    log("  ✓ SLOs passed")
                return ok

    log("No Aggregated row in CSV")
    return False


def wait_for_health(url: str, timeout: int = 120) -> bool:
    """Poll /health until it returns 200."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_load_test(args: argparse.Namespace) -> int:
    global FAILED
    procs: list[subprocess.Popen] = []

    try:
        # --- Start coordinator ---
        log("Starting coordinator on port 50050 ...")
        coordinator_cmd = [
            sys.executable, "-m", "distllm.core.coordinator",
            "--model", args.model,
            "--port", "50050",
            "--dtype", "float32",
            "--nodes", "localhost:50051:0:3", "localhost:50052:4:7",
            "--total-layers", str(TOTAL_LAYERS),
        ]
        proc_coord = subprocess.Popen(
            coordinator_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        procs.append(proc_coord)
        time.sleep(5)

        # --- Start node 0 (layers 0-3) ---
        log("Starting node_0 on port 50051 (layers 0-3) ...")
        node0_cmd = [
            sys.executable, "-m", "distllm.core.node",
            "--node-id", "node_0",
            "--model", args.model,
            "--start-layer", "0",
            "--end-layer", "3",
            "--total-layers", str(TOTAL_LAYERS),
            "--port", "50051",
            "--coordinator-host", "localhost",
            "--coordinator-port", "50050",
            "--device", "cpu",
            "--dtype", "float32",
        ]
        proc_node0 = subprocess.Popen(
            node0_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        procs.append(proc_node0)
        time.sleep(5)

        # --- Start node 1 (layers 4-7) ---
        log("Starting node_1 on port 50052 (layers 4-7) ...")
        node1_cmd = [
            sys.executable, "-m", "distllm.core.node",
            "--node-id", "node_1",
            "--model", args.model,
            "--start-layer", "4",
            "--end-layer", "7",
            "--total-layers", str(TOTAL_LAYERS),
            "--port", "50052",
            "--coordinator-host", "localhost",
            "--coordinator-port", "50050",
            "--device", "cpu",
            "--dtype", "float32",
        ]
        proc_node1 = subprocess.Popen(
            node1_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        procs.append(proc_node1)
        time.sleep(10)

        # --- Start API server ---
        log("Starting API server on port 8000 ...")
        api_cmd = [
            sys.executable, "-m", "distllm.api.server",
            "--port", "8000",
        ]
        proc_api = subprocess.Popen(
            api_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        procs.append(proc_api)

        # --- Wait for API to become healthy ---
        log("Waiting for API server to be healthy ...")
        if not wait_for_health("http://localhost:8000", timeout=args.health_timeout):
            log("✗ API server failed to become healthy")
            return 1
        log("✓ API server is healthy")

        # --- Run Locust ---
        log(f"Starting Locust: {args.users} users, ramp {args.spawn_rate}/s, run {args.run_time}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_prefix = str(RESULTS_DIR / "load_test")

        locust_cmd = [
            sys.executable, "-m", "locust",
            "-f", str(LOCUST_SCENARIO),
            "--headless",
            "-u", str(args.users),
            "-r", str(args.spawn_rate),
            "--run-time", args.run_time,
            "--host", f"http://localhost:{args.api_port}",
            "--csv", csv_prefix,
            "--csv-full-body",
            "--stop-timeout", "10",
        ]
        result = subprocess.run(locust_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"Locust exited with code {result.returncode}")
            for line in result.stderr.split("\n")[-10:]:
                if line.strip():
                    log(f"  {line.strip()}")
        else:
            log("✓ Locust completed")

        # --- SLO check ---
        csv_stats = f"{csv_prefix}_stats.csv"
        if os.path.exists(csv_stats):
            passed = check_slo(csv_stats, args.p95_ms, args.error_rate)
            if not passed:
                log("✗ SLO check FAILED")
                FAILED = True
                return 1
        else:
            log(f"Locust CSV not found at {csv_stats}")
            log("Locust stderr:")
            for line in result.stderr.split("\n")[-20:]:
                if line.strip():
                    log(f"  {line.strip()}")
            return 1

        log("✓ Load test passed")
        return 0

    except KeyboardInterrupt:
        log("Interrupted, shutting down ...")
        return 1
    finally:
        # --- Tear down ---
        log("Shutting down processes ...")
        for p in reversed(procs):
            try:
                os.kill(p.pid, signal.SIGTERM)
            except Exception:
                pass
        for p in reversed(procs):
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster load test with Locust")
    parser.add_argument("--model", default=MODEL, help="HF model name")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--spawn-rate", type=float, default=5)
    parser.add_argument("--run-time", default="3m")
    parser.add_argument("--p95-ms", type=float, default=5000.0)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--health-timeout", type=int, default=180)
    args = parser.parse_args()

    rc = run_load_test(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()

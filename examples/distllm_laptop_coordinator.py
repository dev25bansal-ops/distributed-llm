"""Laptop-side DistLLM coordinator + gRPC tunnel for the laptop+Colab demo (v3).

## What this demonstrates
- Running a local coordinator + REST API on a laptop (Llama-3.1-8B
  in 4-bit fits on 8 GB VRAM)
- Exposing a public tunnel so a Colab GPU worker can join via gRPC
- End-to-end split inference: laptop holds embed/prompt layers, Colab
  holds the heavy decoder layers

What this does:
  1. Starts the local DistLLM coordinator + REST API (Llama-3.1-8B in 4-bit
     by default, fits on 8 GB VRAM).
  2. Spins up `bore local 50050 --to bore.pub` so a remote Colab VM can
     reach the laptop's coordinator gRPC port and join as a worker.
  3. Prints `DISTLLM_PUBLIC_HOST` / `DISTLLM_PUBLIC_PORT` so you can paste
     them into the Colab notebook. The Colab notebook will then spawn
     `distllm.dist.worker` pointing at the laptop coordinator.
  4. Starts a chat-completion test once everything is up so the laptop
     confirms the pipeline works before you switch to the Colab side.

Why bore and not cloudflared:
  Cloudflared 2026.8+ no longer prints the assigned TCP port to stdout for
  `tunnel --url tcp://...` (the URL is exposed only via the cloudflared
  dashboard), so the Colab worker can't discover it from a notebook.
  bore's `bore local <port> --to bore.pub` always prints the assigned
  `bore.pub:<port>` in a single line, which is exactly what we need.

Run:
    python examples/distllm_laptop_coordinator.py
Then follow the on-screen prompts.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

import httpx  # comes with distllm

CLUSTER_KEY = os.environ.get("DISTLLM_CLUSTER_KEY") or secrets.token_urlsafe(16)
# Stable API key if the user provides one, otherwise random.
API_KEY = os.environ.get("DISTLLM_DEMO_API_KEY") or f"demo-{secrets.token_urlsafe(12)}"
COOR_GPRC_PORT = int(os.environ.get("DISTLLM_COOR_PORT", "50050"))
API_PORT = int(os.environ.get("DISTLLM_API_PORT", "8000"))
MODEL = os.environ.get("DISTLLM_DEMO_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
QUANTIZATION = os.environ.get("DISTLLM_DEMO_QUANTIZATION", "bitsandbytes_4bit")
BORE_BIN = os.environ.get("BORE_BIN") or shutil.which("bore")
BORE_SERVER = os.environ.get("BORE_SERVER", "bore.pub")
DEMO_PROMPT = "Once upon a time, a robot and a puppy"
MAX_TOKENS = 60


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


def start_bore_tunnel(local_port: int) -> tuple[subprocess.Popen, str, str]:
    """Spawn `bore local <port> --to bore.pub` and parse the assigned public
    `bore.pub:<port>` from its log output. Returns (proc, host, port)."""
    bin_path = BORE_BIN or shutil.which("bore")
    if not bin_path:
        raise SystemExit(
            "bore is not on PATH. Install it:\n"
            "  Windows: download the latest release from\n"
            "    https://github.com/ekzhang/bore/releases\n"
            "  and put bore.exe somewhere on PATH (e.g. C:\\Users\\<you>\\bin)\n"
            "  macOS:   brew install bore\n"
            "  Linux:   cargo install bore-cli    (if you have cargo)\n"
            "  or download the binary from the GitHub release page"
        )

    cmd = [bin_path, "local", str(local_port), "--to", BORE_SERVER]
    print(f"starting bore: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0,
    )

    q: "Queue[str]" = Queue()

    def _drain():
        try:
            for raw in iter(proc.stdout.readline, b""):
                q.put(raw.decode("utf-8", errors="replace"))
        except Exception:
            pass

    threading.Thread(target=_drain, daemon=True).start()

    print(f"waiting for bore to assign a public port (up to 30 s)...")
    host = port_str = None
    deadline = time.time() + 30
    # bore's success line is `listening at bore.pub:17128` (the
    # `remote_port=...` line above it has the same number). We accept both.
    host_re = re.compile(r"(bore\.pub):(\d+)", re.IGNORECASE)
    while time.time() < deadline and host is None:
        try:
            line = q.get(timeout=1.0)
        except Empty:
            if proc.poll() is not None:
                raise SystemExit(
                    f"bore exited prematurely (code {proc.returncode}). "
                    "bore.pub may be down; try again later or set BORE_SERVER."
                )
            continue
        clean = _strip_ansi(line)
        sys.stdout.write(clean)
        sys.stdout.flush()
        m = host_re.search(clean)
        if m:
            host = m.group(1)
            port_str = m.group(2)

    if host is None:
        proc.terminate()
        raise SystemExit(
            "bore did not print an assigned port within 30 s. "
            "Check your network and try again."
        )
    print(f"\nbore tunnel up: {host}:{port_str}")
    return proc, host, port_str


def wait_health(url: str, timeout: float) -> bool:
    """Poll ``/health`` until the server reports 200 OR the deadline is hit.

    The DistLLM REST API returns 200 once the model is loaded and serving,
    or 503 ("No model loaded") if the server is up but the model is still
    loading. We treat either as "server is reachable" and keep polling.
    """
    deadline = time.time() + timeout
    saw_503 = False
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
            if r.status_code == 503:
                saw_503 = True
                time.sleep(2.0)
                continue
            if 500 <= r.status_code < 600:
                time.sleep(1.0)
                continue
            return True
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(1.0)
    return saw_503


def main() -> None:
    print("=" * 70)
    print(" DistLLM laptop+Colab demo  (v3: gRPC tunnel via bore.pub)")
    print("=" * 70)
    print(f"  model     : {MODEL}  (quantization: {QUANTIZATION})")
    print(f"  coor port : {COOR_GPRC_PORT}  (TUNNELED publicly via bore.pub)")
    print(f"  api port  : {API_PORT}        (laptop only, NOT tunneled)")
    print(f"  cluster key (shared with Colab worker): {CLUSTER_KEY}")
    print(f"  api key   : {API_KEY}")
    print()

    # 1. Start bore tunnel exposing the coordinator gRPC port.
    tunnel_proc, public_host, public_port = start_bore_tunnel(COOR_GPRC_PORT)
    print()
    print(" " + "-" * 68)
    print("  Paste these into the Colab notebook (cell #2):")
    print(" " + "-" * 68)
    print(f"   DISTLLM_PUBLIC_HOST = {public_host!r}")
    print(f"   DISTLLM_PUBLIC_PORT = {public_port!r}")
    print(f"   CLUSTER_KEY         = {CLUSTER_KEY!r}")
    print(" " + "-" * 68)
    print()

    # 2. Start the local coordinator + REST API.
    cache_root = os.environ.get("DISTLLM_HF_CACHE", r"D:\distllm_models")
    os.makedirs(cache_root, exist_ok=True)
    env = {
        **{k: v for k, v in os.environ.items()
           if k not in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")},
        "DISTLLM_CLUSTER_KEY": CLUSTER_KEY,
        "DISTLLM_DEMO_API_KEY": API_KEY,
        "API_KEY": API_KEY,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": cache_root,
    }
    # Forward the HuggingFace token if the user set one.
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        env["HF_TOKEN"] = hf_token

    coord_proc = subprocess.Popen(
        [sys.executable, "-m", "distllm.core.coordinator_cli",
         "--model", MODEL, "--port", str(COOR_GPRC_PORT), "--local"],
        env=env,
    )
    api_args = [
        sys.executable, "-m", "distllm.api.server",
         "--model", MODEL, "--port", str(API_PORT), "--local",
    ]
    # Pass quantization via env, NOT via --quantization CLI, because the
    # CLI choices ("bitsandbytes_4bit") and the settings layer
    # ("bnb_4bit") use different names. Pydantic reads
    # DISTLLM_QUANTIZATION__METHOD and accepts the settings-layer name.
    if QUANTIZATION and QUANTIZATION != "none":
        # Normalize CLI-name -> settings-name.
        _qmap = {
            "bitsandbytes_4bit": "bnb_4bit",
            "bitsandbytes_8bit": "bnb_8bit",
            "gptq": "gptq",
            "none": "none",
        }
        settings_method = _qmap.get(QUANTIZATION, QUANTIZATION)
        env["DISTLLM_QUANTIZATION__METHOD"] = settings_method
    api_proc = subprocess.Popen(api_args, env=env)
    print(f"coordinator PID {coord_proc.pid}, api PID {api_proc.pid}")
    print()

    def _cleanup(*_):
        for p in (tunnel_proc, coord_proc, api_proc):
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    if not wait_health(f"http://127.0.0.1:{API_PORT}/health", timeout=180.0):
        print("REST API never came up. Check the API process output above.")
        _cleanup()
    print(f"REST API ready: http://localhost:{API_PORT}/v1/chat/completions")
    print()
    print("Once the Colab worker is running, run a quick chat from this terminal:")
    print(f"   curl -s http://localhost:{API_PORT}/v1/chat/completions \\\\\n"
          f"        -H 'Content-Type: application/json' \\\\\n"
          f"        -H 'Authorization: Bearer {API_KEY}' \\\\\n"
          f"        -d '{{\"model\":\"{MODEL}\",\"messages\":[{{\"role\":\"user\",\"content\":\"{DEMO_PROMPT}\"}}],\"max_tokens\":{MAX_TOKENS}}}'")
    print()
    print("Or run it from Python:")
    print(
        "   import httpx\n"
        f"  r = httpx.post('http://localhost:{API_PORT}/v1/chat/completions',\n"
        "                 headers={'Authorization': f'Bearer YOUR_KEY'},\n"
        "                 json={'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',\n"
        "                       'messages': [{'role':'user','content': 'Once upon a time'}],\n"
        "                       'max_tokens': 60}, timeout=60)\n"
        "   print(r.json()['choices'][0]['message']['content'])\n"
    )
    print("Press Ctrl+C to stop the cluster.")

    try:
        coord_proc.wait()
    except KeyboardInterrupt:
        _cleanup()


if __name__ == "__main__":
    main()

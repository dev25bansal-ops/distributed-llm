"""Launch gate 2 + 5: prove the index install works for a stranger.

Creates a throwaway venv, installs `distributed-llm` from PyPI (not a local
path), and checks the things a new user is told to do. Any failure here is a
launch blocker, so it prints exactly which step broke.

Run:  python scripts/launch_gate_index_install.py [--spec "distributed-llm[cli]"]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

STEPS = [
    ("import the package", [sys.executable, "-c", "import distllm; print(distllm.__version__)"]),
    ("CLI is on PATH",      [None, "--help"]),          # replaced with the venv's own script
    ("module entry point",  [sys.executable, "-m", "distllm.cli.main", "--help"]),
    ("openai-compatible import", [sys.executable, "-c", "from distllm.api.server import app; print('ok')"]),
]


def _run(label: str, cmd: list[str], env: dict, cwd: str) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=cwd)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after 300s"
    if p.returncode == 0:
        return True, (p.stdout.strip().splitlines() or [""])[-1][:100]
    err = (p.stderr.strip().splitlines() or ["?"])[-1]
    return False, err[:200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="distributed-llm", help="pip requirement to install")
    ap.add_argument("--keep", action="store_true", help="keep the venv for inspection")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="distllm-gate2-")
    vdir = Path(tmp) / "venv"
    print(f"creating venv in {vdir}")
    venv.EnvBuilder(with_pip=True, clear=True).create(vdir)

    bindir = vdir / ("Scripts" if os.name == "nt" else "bin")
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "VIRTUAL_ENV": str(vdir)}
    env.pop("PYTHONPATH", None)  # a stranger has no PYTHONPATH pointing at our src

    print(f"installing {args.spec} from the index ...")
    ok, detail = _run("install", [str(py), "-m", "pip", "install", "--no-cache-dir", args.spec], env, tmp)
    if not ok:
        print(f"FAIL install: {detail}")
        return 1
    print(f"  ok install ({detail})")

    failures = 0
    for label, cmd in STEPS:
        if cmd[0] is None:
            script = bindir / ("distllm.exe" if os.name == "nt" else "distllm")
            if not script.exists():
                print(f"FAIL {label}: no distllm script at {script}")
                failures += 1
                continue
            cmd = [str(script)] + cmd[1:]
        else:
            cmd = [str(py) if c is sys.executable else c for c in cmd]
        ok, detail = _run(label, cmd, env, tmp)
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {detail}")
        failures += 0 if ok else 1

    # The version the index actually serves, for the record.
    import json
    import urllib.request
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/distributed-llm/json", timeout=30) as r:
            idx = json.load(r)["info"]["version"]
        print(f"\nindex serves: {idx}")
    except Exception as e:  # noqa: BLE001 - network probe, informational
        print(f"\nindex probe failed: {type(e).__name__}")

    if args.keep:
        print(f"venv kept at {vdir}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} step(s))'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

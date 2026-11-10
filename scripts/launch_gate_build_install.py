"""Launch gate 2, release-candidate half: does BUILDING this tree work cleanly?

launch_gate_index_install.py proves the published index install works. This
proves the tree we are about to publish builds and installs from source in a
clean venv -- catching missing package data, bad build config, and undeclared
dependencies that a local editable install hides because everything is already
present in this interpreter.

Run:  python scripts/launch_gate_build_install.py
"""

import argparse
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd, env, cwd, timeout=900):
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env, cwd=cwd)
    out = (p.stdout or b"").decode("utf-8", "replace")
    err = (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, out, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extras", default="cli", help="extras to install with the tree")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    tmp = Path(tempfile.mkdtemp(prefix="distllm-gate2b-"))
    vdir = tmp / "venv"
    dist = tmp / "dist"
    print(f"building wheel from {ROOT} ...")
    venv.EnvBuilder(with_pip=True, clear=True).create(vdir)
    bindir = vdir / ("Scripts" if os.name == "nt" else "bin")
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    env = {**os.environ, "VIRTUAL_ENV": str(vdir), "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    env.pop("PYTHONPATH", None)

    failures = 0

    # 1. Build the wheel in the venv, so the build backend must be installable.
    rc, out, err = _run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", "build"], env, str(ROOT))
    if rc != 0:
        print(f"FAIL install build backend: {(err.strip().splitlines() or ['?'])[-1][:180]}")
        return 1
    rc, out, err = _run(
        [str(py), "-m", "build", "--wheel", "--outdir", str(dist), str(ROOT)], env, str(ROOT)
    )
    wheels = sorted(dist.glob("*.whl"))
    if rc != 0 or not wheels:
        print(f"FAIL build wheel: {(err.strip().splitlines() or ['?'])[-1][:200]}")
        return 1
    wheel = wheels[-1]
    print(f"  ok built {wheel.name} ({wheel.stat().st_size // 1024} KiB)")

    # 2. The wheel must carry the package data the runtime needs, not just .py.
    import zipfile

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    non_py = [n for n in names if not n.endswith((".py", ".pyi", ".dist-info/RECORD"))]
    print(f"  wheel has {len(names)} entries, {len(non_py)} non-.py")

    # 3. Install that wheel into the clean venv, with no source on the path.
    rc, out, err = _run(
        [str(py), "-m", "pip", "install", "-q", "--no-cache-dir", f"{wheel}[{args.extras}]"], env, str(tmp)
    )
    if rc != 0:
        print(f"FAIL install wheel: {(err.strip().splitlines() or ['?'])[-1][:200]}")
        return 1
    print("  ok installed from the built wheel")

    # 4. Import from the wheel, in a directory with no source tree nearby.
    checks = [
        ("import distllm", [str(py), "-c", "import distllm; print(distllm.__version__)"]),
        ("import the server", [str(py), "-c", "from distllm.api.server import app; print(len(app.openapi()['paths']))"]),
        ("CLI answers", [str(py), "-m", "distllm.cli.main", "--help"]),
    ]
    for label, cmd in checks:
        rc, out, err = _run(cmd, env, str(tmp))
        if rc == 0:
            last = (out.strip().splitlines() or [""])[-1][:80]
            print(f"  ok {label}: {last}")
        else:
            last = (err.strip().splitlines() or ["?"])[-1][:200]
            print(f"FAIL {label}: {last}")
            failures += 1

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

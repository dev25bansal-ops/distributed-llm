"""CI import-assertion: fail when the measured import graph is truncated.

Companion to the ``testing`` extra (see pyproject.toml). CI installs
``.[dev,testing]``; if any non-optional module fails to import (missing
dependency), coverage is silently computed over a truncated graph and the
coverage number becomes misleading. This script imports every module under
``src/distllm`` and exits non-zero on any failure outside the documented
optional-dependency allowlist.

Usage: python scripts/ci/import_assert.py [--package distllm]
"""

from __future__ import annotations

import argparse
import importlib
import io
import pkgutil
import sys
from pathlib import Path

# Modules allowed to fail: they require optional, heavy, or platform-bound
# dependencies that the testing extra intentionally does not install.
ALLOWED_FAILURES: dict[str, str] = {
    # vLLM backend needs the vllm package (Linux + CUDA, multi-GB)
    "distllm.backends.vllm_backend": "vllm",
    "distllm.backends.nim_backend": "vllm",
    # triton kernel path needs triton
    "distllm.backends.triton_backend": "triton",
}


def iter_modules(package: str):
    yield package
    pkg = importlib.import_module(package)
    path = getattr(pkg, "__path__", None)
    if not path:
        return
    for info in pkgutil.walk_packages(path, prefix=f"{package}."):
        if info.name.split(".")[-1].startswith("_test"):
            continue
        yield info.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="distllm")
    args = parser.parse_args()

    failures: list[tuple[str, str, str]] = []
    count = 0
    for name in iter_modules(args.package):
        count += 1
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - report everything
            reason = type(exc).__name__
            detail = str(exc).strip().splitlines()[0] if str(exc) else ""
            if name in ALLOWED_FAILURES:
                continue
            failures.append((name, reason, detail))

    if failures:
        print(f"IMPORT ASSERTION FAILED: {len(failures)} module(s) failed to import")
        for name, reason, detail in sorted(failures):
            print(f"  {name}: {reason}: {detail}")
        print(
            "\nIf a failure is caused by an optional dependency, either add the "
            "dep to the 'testing' extra (pyproject.toml) or add an explicit "
            "entry to ALLOWED_FAILURES in scripts/ci/import_assert.py."
        )
        return 1

    print(f"import assertion OK: {count} modules imported cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

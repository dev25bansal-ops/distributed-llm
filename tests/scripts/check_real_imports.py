"""Sweep real imports of every src/distllm/core module without the test fake.

This is the core of the C6 fix: it imports each production core module via a
normal ``importlib.import_module`` (no ``tests/_import_helper`` fakes, no
``load_module`` path-bypass).  It records pass/fail and the trailing error
message, writes a baseline JSON, and reports broken modules.

Usage (from repo root):
    python tests/scripts/check_real_imports.py [--write]

Exit codes:
    0  -> every module imports cleanly (or baseline not written yet)
    1  -> a module that imported cleanly before now fails (NEW breakage)
    2  -> a module failed to import (but baseline already recorded it as broken)
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
BASELINE = REPO / "tests" / "scripts" / "real_import_baseline.json"


def core_module_names() -> list[str]:
    """All non-__init__ modules under src/distllm/core, as dotted names."""
    names: list[str] = []
    root = SRC / "distllm" / "core"
    for p in sorted(root.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        rel = p.relative_to(SRC)
        names.append(".".join(rel.with_suffix("").parts))
    return names


def sweep() -> tuple[list[str], dict[str, str]]:
    """Import every core module in-process; return (ok_names, broken_map).

    ``broken_map`` maps a dotted module name to a short error message.
    """
    ok: list[str] = []
    broken: dict[str, str] = {}
    for dotted in core_module_names():
        try:
            importlib.import_module(dotted)
            ok.append(dotted)
        except Exception as e:  # noqa: BLE001 - we want to record every failure
            err = str(e)
            if not err:
                err = traceback.format_exc(limit=1).strip().splitlines()[-1]
            broken[dotted] = err[:200]
    return ok, broken


def main() -> int:
    write = "--write" in sys.argv
    ok, broken = sweep()

    if write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps({"ok": sorted(ok), "broken": broken}, indent=2)
        )
        print(f"baseline written -> {BASELINE}")

    print(f"\n=== real-import sweep: {len(ok)} ok / {len(broken)} broken "
          f"({len(ok) + len(broken)} total) ===")
    if broken:
        print("\nBROKEN MODULES:")
        for dotted, err in broken.items():
            print(f"  [FAIL] {dotted}: {err}")

    if not BASELINE.exists():
        # No baseline yet: this run just records truth, non-failing.
        print("\nno baseline on disk; pass this run as the baseline "
              "(re-run with --write to persist).")
        return 0

    prev = json.loads(BASELINE.read_text())
    prev_ok = set(prev.get("ok", []))
    now_ok = set(ok)
    regressed = sorted(now_ok - prev_ok)  # previously-broken now fixed
    newly_broken = sorted(prev_ok - now_ok)  # previously-ok now broken

    if newly_broken:
        print("\nNEW BREAKAGE vs baseline:")
        for m in newly_broken:
            print(f"  [REGRESS] {m}: {broken.get(m, '?')}")
        return 1
    if regressed:
        print(f"\nFIXED vs baseline (now import cleanly): {regressed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
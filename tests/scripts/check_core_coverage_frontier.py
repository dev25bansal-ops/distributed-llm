"""Coverage-frontier checker for src/distllm/core (audit finding C7).

A core module is considered COVERED if any test file under ``tests/``
references it, by:
  * a dotted import (``import distllm.core.X`` / ``from distllm.core.X import …``),
  * ``load_module("distllm/core/X.py")``,
  * ``spec_from_file_location(…, "…distllm/core/X.py")``.

The script reports the modules with ZERO test references, grouped by
subsystem, and compares against a recorded baseline
(``tests/scripts/core_coverage_frontier.json``).  It exits non-zero when the
zero-coverage set GROWS (new modules added without tests) — existing debt is
reported, not gated.

Usage:  python tests/scripts/check_core_coverage_frontier.py [--write]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
TESTS = REPO / "tests"
BASELINE = REPO / "tests" / "scripts" / "core_coverage_frontier.json"


def core_modules() -> list[tuple[str, str]]:
    """Return (dotted_name, rel_path) for every module under core."""
    out: list[tuple[str, str]] = []
    root = SRC / "distllm" / "core"
    for p in sorted(root.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        rel = p.relative_to(SRC)
        dotted = ".".join(rel.with_suffix("").parts)
        out.append((dotted, rel.as_posix()))
    return out


def test_files() -> list[Path]:
    return sorted(p for p in TESTS.rglob("*.py") if p.name != "__init__.py")


def reference_tokens(dotted: str, rel_path: str) -> list[str]:
    """Substrings that indicate a test references this core module."""
    return [
        dotted,                        # import distllm.core.X
        rel_path,                      # load_module("distllm/core/X.py")
        rel_path.removesuffix(".py"),  # spec name "distllm.core.X"
        rel_path.replace("/", "\\"),   # Windows paths
    ]


def find_zero_coverage() -> dict[str, str]:
    files = test_files()
    texts = [f.read_text(encoding="utf-8", errors="ignore") for f in files]
    uncovered: dict[str, str] = {}
    for dotted, rel in core_modules():
        tokens = reference_tokens(dotted, rel)
        if not any(tok in text for text in texts for tok in tokens):
            uncovered[dotted] = rel
    return uncovered


def group_by_subsystem(uncovered: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for dotted in uncovered:
        rel = uncovered[dotted]
        parts = rel.removeprefix("distllm/core/").split("/")
        group = parts[0] if len(parts) > 1 else "core"
        groups.setdefault(group, []).append(dotted)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def main() -> int:
    write = "--write" in sys.argv
    uncovered = find_zero_coverage()
    groups = group_by_subsystem(uncovered)
    count = len(uncovered)

    print(f"\n=== core coverage frontier: {count} modules with zero test refs ===")
    for group, mods in groups.items():
        print(f"\n[{group}] ({len(mods)})")
        for m in mods:
            print(f"  {m}")

    if write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({"zero_coverage": sorted(uncovered)}, indent=2))
        print(f"\nbaseline written -> {BASELINE}")

    if not BASELINE.exists():
        print("\nno baseline on disk; re-run with --write to record it.")
        return 0

    prev = set(json.loads(BASELINE.read_text()).get("zero_coverage", []))
    now = set(uncovered)
    grown = sorted(now - prev)
    shrunk = sorted(prev - now)
    if grown:
        print(f"\nNEW zero-coverage modules (cover or remove these): {grown}")
        return 1
    if shrunk:
        print(f"\nCOVERED since baseline (great): {shrunk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
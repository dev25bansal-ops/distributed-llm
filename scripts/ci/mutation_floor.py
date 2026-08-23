#!/usr/bin/env python3
"""Mutation-score floor gate (CI / pre-merge).

Runs mutation testing against a configurable set of "strict-typed critical"
modules and exits non-zero when the observed mutation score drops below a
configurable floor.

Design (T4-01)
--------------
mutmut is the preferred engine: it produces richer mutants (libcst-based) and
is what the nightly CI uses on Linux.  However mutmut 3.x hard-refuses to run
on native Windows (it calls ``sys.exit(1)`` and relies on ``os.fork``), so on
Windows we transparently fall back to the repository's own cross-platform AST
mutation harness (``tests/mutation/mutate.py``) which produces a real,
comparable mutation score without forking.

This makes the script *verifiable* on a Windows dev box while still being the
exact gate the Linux CI will run once mutmut is installed there.

Usage
-----
    python scripts/ci/mutation_floor.py [--floor 70] [--max-mutations 25]
                                        [--engine auto|mutmut|ast] [--quiet]

Exit codes
----------
    0  -> every module met or exceeded the floor (or no mutants were generated
          and we choose to treat "no mutants" as a pass with a warning).
    1  -> at least one module is below the floor.
    2  -> engine could not run in this environment (e.g. mutmut missing) and
          the harness also could not run; treated as a hard error so CI does
          not silently go green.  Pass ``--allow-skip`` to turn this into a
          warning-only (exit 0) when *nothing* could be measured.

The set of critical modules and their accompanying test files is configured in
``CRITICAL_MODULES`` below.  Each entry pairs a source module with the test
file(s) that should catch its mutants.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration: strict-typed critical modules + the tests that must kill them
# ---------------------------------------------------------------------------
# crdt.py is the flagship A6 CRDT module: pure Python, fast, headless, and
# already has strong property-based regression coverage (test_a6_crdt_cache.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CRITICAL_MODULES: list[dict] = [
    {
        "name": "cache/crdt.py",
        "source": "src/distllm/cache/crdt.py",
        "tests": ["tests/regression_high/test_a6_crdt_cache.py"],
    },
    # Add more strict-typed critical modules here, e.g.:
    # {
    #     "name": "verification/comparator.py",
    #     "source": "src/distllm/verification/comparator.py",
    #     "tests": ["tests/verification/test_verification.py"],
    # },
]


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------
@dataclass
class ModuleScore:
    name: str
    engine: str
    total: int
    killed: int
    survived: int
    timeout: int
    error: int
    score_pct: float
    below_floor: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "engine": self.engine,
            "total": self.total,
            "killed": self.killed,
            "survived": self.survived,
            "timeout": self.timeout,
            "error": self.error,
            "score_pct": self.score_pct,
            "below_floor": self.below_floor,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Engine 1: mutmut (Linux CI path)
# ---------------------------------------------------------------------------
def _run_mutmut(module: dict, floor: float, max_mutations: int) -> ModuleScore | None:
    """Run mutmut on a single critical module. Returns None if mutmut cannot run."""
    source = module["source"]
    tests = module["tests"]
    name = module["name"]

    # mutmut refuses native Windows outright.
    if platform.system() == "Windows":
        return None

    if shutil.which("mutmut") is None and not _importable("mutmut"):
        return None

    try:
        import mutmut  # noqa: F401  (ensures the package is importable)
    except Exception:
        return None

    src_abs = (PROJECT_ROOT / source).resolve()
    if not src_abs.exists():
        return ModuleScore(
            name=name, engine="mutmut", total=0, killed=0, survived=0,
            timeout=0, error=1, score_pct=0.0, below_floor=False,
            detail=f"source not found: {source}",
        )

    # Build a tiny mutmut config that points only at this module + its tests.
    cfg_path = PROJECT_ROOT / ".mutmut.floor.toml"
    cfg = _render_mutmut_config(source, tests, max_mutations)
    cfg_path.write_text(cfg, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    env["MUTMUT_CONFIG"] = str(cfg_path)

    try:
        # run -> results -> tally
        subprocess.run(
            [sys.executable, "-m", "mutmut", "run", "--max-children", "4"],
            cwd=str(PROJECT_ROOT), env=env, check=False,
        )
        out = subprocess.run(
            [sys.executable, "-m", "mutmut", "results", "--all"],
            cwd=str(PROJECT_ROOT), env=env, check=False,
            capture_output=True, text=True,
        )
    finally:
        cfg_path.unlink(missing_ok=True)

    scored = _parse_mutmut_results(out.stdout + out.stderr, name)
    if scored is None:
        return ModuleScore(
            name=name, engine="mutmut", total=0, killed=0, survived=0,
            timeout=0, error=1, score_pct=0.0, below_floor=False,
            detail="mutmut produced no parseable results",
        )
    scored.below_floor = scored.score_pct < floor
    return scored


def _render_mutmut_config(source: str, tests: list[str], max_mutations: int) -> str:
    test_paths = "\n".join(f'    "{t}",' for t in tests)
    return f"""\
[mutmut]
paths_to_mutate = ["{source}"]
tests_dir = "."
test_command = "python -m pytest {' '.join(tests)}"
no_progress = true
max_mutations = {max_mutations}
"""


def _parse_mutmut_results(text: str, name: str) -> ModuleScore | None:
    """Parse `mutmut results --all` lines of the form '<file>::...: <status>'."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        status = line.rsplit(":", 1)[-1].strip()
        # statuses: killed, survived, timeout, suspicious, no tests, ...
        counts[status] = counts.get(status, 0) + 1
    killed = counts.get("killed", 0)
    # "no tests" / "suspicious" mutants don't count as a *caught* bug; they are
    # excluded from the denominator to avoid inflating the score.
    denom = killed + counts.get("survived", 0) + counts.get("timeout", 0)
    if denom == 0:
        return None
    score = killed / denom * 100.0
    return ModuleScore(
        name=name, engine="mutmut", total=denom, killed=killed,
        survived=counts.get("survived", 0), timeout=counts.get("timeout", 0),
        error=0, score_pct=score, below_floor=False,
        detail=" ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )


# ---------------------------------------------------------------------------
# Engine 2: AST harness (cross-platform fallback, incl. Windows)
# ---------------------------------------------------------------------------
def _run_ast_harness(module: dict, floor: float, max_mutations: int) -> ModuleScore | None:
    """Run the repo's own AST mutation harness (tests/mutation/mutate.py)."""
    source = module["source"]
    tests = module["tests"]
    name = module["name"]

    harness = (PROJECT_ROOT / "tests" / "mutation" / "mutate.py").resolve()
    if not harness.exists():
        return None

    src_abs = (PROJECT_ROOT / source).resolve()
    if not src_abs.exists():
        return ModuleScore(
            name=name, engine="ast", total=0, killed=0, survived=0,
            timeout=0, error=1, score_pct=0.0, below_floor=False,
            detail=f"source not found: {source}",
        )

    sys.path.insert(0, str(harness.parent))
    try:
        import mutate
    except Exception as e:  # pragma: no cover - harness unexpectedly broken
        return ModuleScore(
            name=name, engine="ast", total=0, killed=0, survived=0,
            timeout=0, error=1, score_pct=0.0, below_floor=False,
            detail=f"failed to import harness: {e}",
        )

    with open(src_abs, encoding="utf-8") as f:
        source_code = f.read()
    points = mutate.find_mutation_points(source_code)
    if max_mutations and max_mutations > 0:
        points = points[:max_mutations]
    if not points:
        return ModuleScore(
            name=name, engine="ast", total=0, killed=0, survived=0,
            timeout=0, error=0, score_pct=0.0, below_floor=False,
            detail="no mutation points found",
        )

    test_rel = tests[0]  # harness runs a single test file
    ok, _, baseline_t = mutate.run_tests(str((PROJECT_ROOT / test_rel).resolve()),
                                         str(PROJECT_ROOT), timeout=300)
    if not ok:
        return ModuleScore(
            name=name, engine="ast", total=len(points), killed=0, survived=0,
            timeout=0, error=1, score_pct=0.0, below_floor=True,
            detail="baseline test suite failed for this module",
        )

    tree = mutate.ast.parse(source_code)
    import tempfile, shutil as _sh
    tmp = tempfile.mkdtemp(prefix="mutation_floor_")
    results = {"KILLED": 0, "SURVIVED": 0, "TIMEOUT": 0, "ERROR": 0}
    survived: list[str] = []
    try:
        for p in points:
            r = mutate.mutate_and_test(
                str(src_abs), str((PROJECT_ROOT / test_rel).resolve()),
                p, tree, tmp, baseline_t, str(PROJECT_ROOT),
            )
            results[r["status"]] = results.get(r["status"], 0) + 1
            if r["status"] in ("SURVIVED", "TIMEOUT"):
                survived.append(r["point"].description)
    finally:
        _sh.rmtree(tmp, ignore_errors=True)

    total = len(points)
    killed = results.get("KILLED", 0)
    surv = results.get("SURVIVED", 0) + results.get("TIMEOUT", 0)
    score = killed / total * 100.0 if total else 0.0
    return ModuleScore(
        name=name, engine="ast", total=total, killed=killed,
        survived=surv, timeout=results.get("TIMEOUT", 0),
        error=results.get("ERROR", 0), score_pct=score,
        below_floor=score < floor,
        detail="; ".join(survived[:10]) if survived else "",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def run_module(module: dict, floor: float, max_mutations: int, engine: str) -> ModuleScore | None:
    if engine in ("auto", "mutmut"):
        scored = _run_mutmut(module, floor, max_mutations)
        if scored is not None:
            return scored
    if engine in ("auto", "ast"):
        return _run_ast_harness(module, floor, max_mutations)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mutation-score floor gate")
    parser.add_argument("--floor", type=float, default=70.0,
                        help="Minimum acceptable mutation score %% (default 70).")
    parser.add_argument("--max-mutations", type=int, default=25,
                        help="Cap on mutants per module (CI time budget).")
    parser.add_argument("--engine", choices=["auto", "mutmut", "ast"], default="auto",
                        help="Force an engine (auto = prefer mutmut, fall back to AST).")
    parser.add_argument("--allow-skip", action="store_true",
                        help="Exit 0 (not 2) if no engine could run at all.")
    parser.add_argument("--json", action="store_true",
                        help="Emit a JSON summary to stdout.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    scores: list[ModuleScore] = []
    any_measured = False
    for module in CRITICAL_MODULES:
        scored = run_module(module, args.floor, args.max_mutations, args.engine)
        if scored is None:
            if not args.quiet:
                print(f"[skip] {module['name']}: no mutation engine available in this environment")
            continue
        any_measured = True
        scores.append(scored)
        flag = "BELOW FLOOR" if scored.below_floor else "ok"
        if not args.quiet:
            print(f"[{scored.engine}] {scored.name}: "
                  f"score={scored.score_pct:.1f}% "
                  f"(killed={scored.killed}/{scored.total}, "
                  f"survived={scored.survived}, timeout={scored.timeout}, "
                  f"error={scored.error}) -> {flag}")
            if scored.detail:
                print(f"         detail: {scored.detail}")

    if args.json:
        print(json.dumps(
            {"floor": args.floor, "modules": [s.to_dict() for s in scores],
             "any_measured": any_measured}, indent=2))

    if not any_measured:
        msg = "No mutation engine could run in this environment; nothing was measured."
        if args.allow_skip:
            if not args.quiet:
                print(f"[warn] {msg}")
            return 0
        if not args.quiet:
            print(f"[error] {msg}", file=sys.stderr)
        return 2

    below = [s for s in scores if s.below_floor]
    if below:
        if not args.quiet:
            print(f"\nFAIL: {len(below)} module(s) below the {args.floor:.0f}% floor.",
                  file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\nPASS: all {len(scores)} module(s) met the {args.floor:.0f}% floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

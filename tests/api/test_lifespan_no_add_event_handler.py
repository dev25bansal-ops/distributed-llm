"""Regression: `app.add_event_handler` was removed in Starlette >=0.40.

This test parses every Python module under `src/distllm/api/` and asserts
that no module makes a live `add_event_handler(...)` call.

The historical site of the bug was `src/distllm/api/server.py` (line 2138
in the previous codebase), where `app.add_event_handler("shutdown", ...)`
was replaced with a `lifespan` async context manager. This test guards
against a re-introduction of the pattern.

Comments mentioning the historical API are allowed; live call sites are not.
We use the AST to detect live call sites — comments and docstrings cannot
match because they never appear as `ast.Call` nodes.
"""

from __future__ import annotations

import ast
from pathlib import Path


API_ROOT = Path("src/distllm/api")


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def _find_add_event_handler_calls(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, code_text) for live `add_event_handler(...)` calls."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_event_handler":
            hits.append((node.lineno, ast.unparse(node)))
    return hits


def test_no_add_event_handler_calls_in_api_modules():
    """No `*.add_event_handler(...)` calls are allowed anywhere under src/distllm/api/.

    The legacy Starlette API was removed in Starlette >=0.40; the supported
    replacement is a `lifespan` async context manager (see server.py).
    """
    assert API_ROOT.is_dir(), f"API_ROOT not found: {API_ROOT}"

    files = _iter_python_files(API_ROOT)
    assert files, "No Python files found under API_ROOT"

    offenders: list[tuple[Path, int, str]] = []
    for f in files:
        hits = _find_add_event_handler_calls(f)
        for lineno, code in hits:
            offenders.append((f, lineno, code))

    assert not offenders, (
        "Found live `add_event_handler(...)` call(s); replace with a "
        "lifespan context manager (Starlette >=0.40 dropped add_event_handler):\n"
        + "\n".join(f"  {p}:{ln}: {c}" for p, ln, c in offenders)
    )


def test_api_directory_walked():
    """Sanity check: the test actually walked the API tree.

    Without this the `test_no_add_event_handler_calls_in_api_modules` test
    could silently pass against an empty directory.
    """
    files = _iter_python_files(API_ROOT)
    assert len(files) >= 20, (
        f"Expected to walk >=20 Python files under {API_ROOT}, found {len(files)}. "
        "If the API tree has shrunk dramatically, investigate before relaxing this bound."
    )


def test_server_lifespan_is_present():
    """Belt-and-braces: the replacement lifespan handler must exist.

    If a future refactor accidentally removes the lifespan wiring while also
    having removed all add_event_handler calls, this regression would still
    pass — but the shutdown path would silently regress. This test catches
    that by asserting the wiring in `server.py` is intact.
    """
    server_py = API_ROOT / "server.py"
    assert server_py.is_file(), f"{server_py} not found"
    src = server_py.read_text(encoding="utf-8")
    assert "lifespan_context" in src, (
        "Expected `lifespan_context` wiring in server.py — did someone remove "
        "the lifespan replacement for add_event_handler?"
    )
    assert "_shutdown" in src, (
        "Expected the `_shutdown` async handler to still be present in server.py"
    )
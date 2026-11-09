"""CQ-08: no silent `except Exception:` in src/distllm.

Every handler body must either log (logger/log), re-raise (raise), or
carry a `# CQ-08`/`noqa` comment naming why swallowing is intentional.
Silent swallows hide the exact failures (auth bypass, data loss) this
repo's audit history is full of.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "distllm"


def _handler_is_documented(node: ast.ExceptHandler) -> bool:
    body_src = ast.unparse(node) if hasattr(ast, "unparse") else ""
    for stmt in node.body:
        if isinstance(stmt, (ast.Expr, ast.Call)):
            # logger.x(...) / log.x(...)
            call = stmt.value if isinstance(stmt, ast.Expr) else stmt
            func = getattr(call, "func", None)
            if isinstance(func, ast.Attribute) and func.attr in {"warning", "error", "info", "debug", "exception"}:
                return True
        if isinstance(stmt, ast.Raise):
            return True
    text = body_src.lower()
    return "cq-08" in text or "noqa" in text


def silent_except_sites() -> list[str]:
    sites = []
    for py in ROOT.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                names = []
                t = node.type
                if isinstance(t, ast.Name):
                    names = [t.id]
                elif isinstance(t, ast.Tuple):
                    names = [e.id for e in t.elts if isinstance(e, ast.Name)]
                if "Exception" not in names:
                    continue
                if not _handler_is_documented(node):
                    sites.append(f"{py}:{node.lineno}")
    return sites


def test_no_silent_except_exception():
    sites = silent_except_sites()
    assert not sites, (
        f"{len(sites)} silent 'except Exception:' handler(s) — add logger/raise "
        f"or a '# CQ-08: <why>' comment: " + ", ".join(sites[:8])
    )


if __name__ == "__main__":
    found = silent_except_sites()
    print(f"{len(found)} silent sites")
    for s in found[:20]:
        print("  ", s)

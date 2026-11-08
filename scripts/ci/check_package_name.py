#!/usr/bin/env python3
"""Reject dependencies on the wrong PyPI project name.

This project ships as ``distributed-llm``. There is an unrelated project on
PyPI called ``distllm`` (1.0.2, "Distributed Inference for Large Language
Models") and ``distllm`` is also the import name here, so writing

    dependencies = ["distllm>=0.1.0"]

looks right and is silently wrong: pip installs the foreign package, which
replaces this project's own install. All ten integrations declared exactly that,
and CI showed it directly:

    Successfully installed ... distllm-1.0.2 ...

after which `import distllm.sdk` raised ModuleNotFoundError -- the foreign
distribution had overwritten the editable install.

This is also a dependency-confusion foothold, so it is checked, not just fixed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, which CI runs
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

WRONG = "distllm"
RIGHT = "distributed-llm"

# Directories that legitimately contain the import name rather than the
# distribution name.
SKIP_PARTS = {"src", "tests", "node_modules", ".git", "build", "dist"}


def dep_names(text: str) -> list[str]:
    if tomllib is None:
        return []
    try:
        data = tomllib.loads(text)
    except Exception:
        return []
    names: list[str] = []
    project = data.get("project") or {}
    for spec in project.get("dependencies") or []:
        names.append(spec)
    for group in (project.get("optional-dependencies") or {}).values():
        names.extend(group)
    return names


def main() -> int:
    root = Path(".")
    offenders: list[tuple[Path, str]] = []

    for path in root.rglob("pyproject.toml"):
        if SKIP_PARTS & set(path.parts):
            continue
        for spec in dep_names(path.read_text(encoding="utf-8")):
            name = re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0].strip()
            if name.lower().replace("_", "-") == WRONG:
                offenders.append((path, spec))

    if offenders:
        print(f'Dependencies on "{WRONG}" found -- the project publishes as "{RIGHT}",')
        print(f'and "{WRONG}" is an unrelated package on PyPI that replaces this one:')
        for path, spec in offenders:
            print(f"  {path}: {spec}")
        return 1

    print(f'OK -- no dependencies on the shadowing package name "{WRONG}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Fail when a core dependency is missing from requirements.lock.

The Docker image installs requirements.lock (not pyproject), so a core
dependency absent from the lock yields an image that builds cleanly and then
dies on first import. That is how transformers, grpcio/protobuf and
python-multipart were each discovered -- one CI run at a time, each costing a
20-minute rebuild. This check makes the whole class visible in seconds.

setuptools is excluded: it is the build tool, installed separately in the
image, and pinning it in the lock would not affect runtime imports.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10, which CI runs
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # neither available: parse the one table we need
        tomllib = None  # type: ignore[assignment]

BUILD_ONLY = {"setuptools"}


def load_dependencies(text: str) -> list[str]:
    """Return ``[project].dependencies`` from pyproject text."""
    if tomllib is not None:
        return tomllib.loads(text)["project"]["dependencies"]
    # Minimal fallback: read the `dependencies = [...]` array in [project].
    match = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    if not match:
        raise SystemExit("could not find [project].dependencies in pyproject.toml")
    return re.findall(r'"([^"]+)"', match.group(1))


def normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def main() -> int:
    proj_text = Path("pyproject.toml").read_text(encoding="utf-8")
    deps = load_dependencies(proj_text)

    lock_text = Path("requirements.lock").read_text(encoding="utf-8").lower()
    locked = {
        normalise(n)
        for n in re.findall(r"^([a-z0-9][a-z0-9_.-]*)\s*(?:==|>=|<=|~=|!=)", lock_text, re.M)
    }

    missing = []
    for dep in deps:
        name = normalise(re.split(r"[<>=!~\[; ]", dep.strip(), maxsplit=1)[0])
        if name and name not in locked and name not in BUILD_ONLY:
            missing.append((name, dep))

    if missing:
        print("Core dependencies missing from requirements.lock:")
        for name, dep in missing:
            print(f"  {name:24s} {dep}")
        print("\nAdd them to requirements.lock, or the Docker image will build")
        print("fine and fail at runtime with ModuleNotFoundError.")
        return 1

    print(f"OK -- all {len(deps)} core dependencies are pinned in requirements.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
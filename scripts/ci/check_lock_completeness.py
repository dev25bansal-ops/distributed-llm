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
import tomllib
from pathlib import Path

BUILD_ONLY = {"setuptools"}


def normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def main() -> int:
    proj = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = proj["project"]["dependencies"]

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
"""Every endpoint path documented in docs/api.md must exist in the real app.

A docs page listing an endpoint a user cannot call is the same defect as a
documented CLI command that does not exist: the reader follows it and gets a 404
with no clue why. Found three today -- gossip paths missing their /api prefix and
a federation heartbeat route that has never existed -- so it is checked now.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOC = ROOT / "docs" / "api.md"
DOCUMENTED = re.compile(r"`(?:GET|POST|PUT|DELETE|PATCH)\s+(/[A-Za-z0-9/_{}.-]+)`")


def main() -> int:
    from distllm.api.server import app

    real = set(app.openapi()["paths"])
    if not DOC.exists():
        print(f"note: {DOC} not found, skipped")
        return 0

    paths = sorted(set(DOCUMENTED.findall(DOC.read_text(encoding="utf-8"))))
    missing = [p for p in paths if p not in real]

    print(f"checked {len(paths)} documented endpoint(s) against {len(real)} real routes")
    if missing:
        print("\nDOCUMENTED BUT NOT ROUTED:")
        for p in missing:
            # Point at the closest real route so the fix is obvious.
            stem = p.rstrip("/").rsplit("/", 1)[0]
            near = sorted(r for r in real if r.startswith(stem))[:3]
            hint = f"  (near: {near})" if near else ""
            print(f"  {p}{hint}")
        return 1

    print("all documented endpoints exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())

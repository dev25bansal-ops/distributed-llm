"""Every kustomize overlay must build with the DEFAULT load restrictor.

ArgoCD and Flux do not pass `--load-restrictor=LoadRestrictionsNone`, so an
overlay that only builds with that flag is not deployable through GitOps. CI
masks this by passing the flag; this check does not.

It also asserts what the build *renders*, not just that it exits 0 -- a worker
StatefulSet above one replica would put two pods on the same layer range, which
core/topology_registry.validate_assignment declares invalid. Both checks exist
because both failed once: an overlay overrode the base back to 2 replicas, and a
resource path above the overlay root made production unbuildable for ArgoCD.

Requires kubectl on PATH. Run from the repo root.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ("dev", "staging", "production")


def main() -> int:
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        print("skipped: kubectl not on PATH")
        return 0

    failures = 0
    for name in OVERLAYS:
        path = ROOT / "deploy" / "kustomize" / name
        if not path.is_dir():
            print(f"FAIL {name}: {path} does not exist")
            failures += 1
            continue
        # No --load-restrictor flag: this is the GitOps condition.
        p = subprocess.run([kubectl, "kustomize", str(path)], capture_output=True, text=True)
        if p.returncode != 0:
            tail = (p.stderr.strip().splitlines() or ["?"])[-1][:180]
            print(f"FAIL {name}: {tail}")
            failures += 1
            continue

        rendered = []
        for doc in p.stdout.split("\n---\n"):
            if "kind: StatefulSet" not in doc:
                continue
            # Parse the object's OWN metadata block, not the first `name:` in
            # the document -- that is the app.kubernetes.io/name label, so a
            # naive search yields "distributed-llm" for every StatefulSet and
            # the worker check below can never match. Verified by injecting
            # replicas: 3 into the staging patch: the naive version printed
            # "ok staging: distributed-llm=3  PASS".
            meta = re.search(r"^metadata:\s*\n((?:[ \t]+.*\n?)+)", doc, re.M)
            block = meta.group(1) if meta else doc
            n = re.search(r"^[ \t]+name:\s*(\S+)", block, re.M)
            r = re.search(r"^\s*replicas:\s*(\d+)", doc, re.M)
            rendered.append((n.group(1) if n else "?", int(r.group(1)) if r else 1))
        print(f"ok   {name}: " + (", ".join(f"{n}={r}" for n, r in rendered) or "no StatefulSets"))

        for n, r in rendered:
            if "worker" in n and r != 1:
                print(f"FAIL {name}: worker {n} renders {r} replicas; one per layer range")
                failures += 1

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

## 2026-11-06 — pip-audit red: transitive filelock + setuptools (FIXED)

- dependency-scan (SEC-4 blocking gate) went red on master AND on PR #10:
  filelock 4.1.0 (17 CVEs, advisory fix 4.2.1 does not exist on PyPI —
  4.1.0 is latest) + setuptools 79.0.1 (PYSEC-2026-3447 fix 83.0.0;
  GHSA-xmj3-9gf7-h52w fix 84.1.0).
- Not a PR regression: master check-run fails identically (verified via
  gh api check-runs on master's latest commit).
- Fix: pyproject.toml floors setuptools>=84.1.0 (covers both advisories).
  filelock is a CI-tooling transitive dep (CacheControl <- pip-audit), not a
  project runtime dep, and no patched release exists upstream, so the 17
  filelock CVEs are waived in .github/workflows/dependency-scan.yml with
  --ignore-vuln, one flag per ID (pip-audit's CLI takes exactly one ID per
  flag), each listed explicitly (no silent waivers), expiry
  2026-12-06: re-check upstream; if 4.2.1 ships, drop the waiver and floor
  filelock>=4.2.1 instead.
- A first attempt (filelock>=4.2.1 floor) was reverted: it broke
  `pip install .` in CI because 4.2.1 does not exist.
- A second attempt (all 17 IDs space-separated on one --ignore-vuln flag)
  died with "unrecognized arguments"; fixed to one flag per ID.

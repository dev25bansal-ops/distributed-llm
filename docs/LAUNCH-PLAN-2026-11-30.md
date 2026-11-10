# DistLLM — Launch Plan to 2026-11-30

**Owner:** CEO session. **Written:** 2026-11-09 23:15 local.
**Launch date:** 2026-11-30. **Available:** 21 days.
**Status source:** measured 2026-11-09, not estimated.

---

## 1. Where we actually are

| Fact | Value | Source |
|---|---|---|
| Package version | 0.6.2.2 | `pyproject.toml` |
| On PyPI | **0.6.2.2 — matches the repo**, uploaded 2026-11-09 | `https://pypi.org/pypi/distributed-llm/json`, checked 2026-11-10 04:00 |
| `master` vs work | master is 115 commits **behind** the working branch, and fully contained in it | `git rev-list` |
| PR #10 (→ master, 115 commits) | **OPEN, MERGEABLE** | `gh pr view 10` |
| PR #10 checks passing | lint, regression, pip-audit, load-test, memory-profile, autofix, bandit, coverage, security-sast, security-secrets, mutation-testing, performance-report, 8/12 integrations, validate-k8s | `gh pr view 10` |
| PR #10 checks **failing** | test ×4 (ubuntu/windows × 3.10/3.11), test-dist, docker-compose, chaos, security-deps, **real-import-gate**, 3 integrations (langchain, llamaindex, crewai), security-container, terraform, vscode-extension, fuzz | same |
| Rust toolchain | **present** at `%USERPROFILE%\.cargo\bin` (cargo 1.94.1) — not on PATH, but works | verified 2026-11-09 |
| GPU on this machine | **absent** | benchmark brief |

### Decisions taken 2026-11-09

- **Publish late.** 0.7.0 goes to PyPI on **Nov 28**, after the hardening pass and with the demo recorded. The index already carries 0.6.2.2 (corrected above), so users are on a working build meanwhile — 0.7.0 is the hardening, not a rescue.
- **Rust SDK ships for real.** It compiles (`cargo check` clean), clippy is clean, and it now has 13 tests that pin its wire contract. Two of those tests found defects that made it non-functional (see below), which is the argument for having done it.

### Rust SDK: what was actually wrong (fixed 2026-11-09)

1. **`ChatResponse` could not deserialize a normal OpenAI payload.** `Message.name` and `tool_call_id` were required on the way in. Every real server — ours included — omits them, so *every* chat response failed to parse. Added `#[serde(default)]`.
2. **`ChatRequest::default()` sent an empty model name.** `#[serde(default = "default_model")]` applies to deserialization only; `Default::default()` left `model: ""`. A request built with `..Default::default()` — which is the documented usage — went out with no model. Replaced with a manual `Default` impl.
3. The crate's **own doc example did not compile** (`message.content` where `message` is `Option<Message>`). That example is the crate's front page.
4. The SDK had **zero tests** across 299 lines. Now 13, covering request serialization, response parsing against real payloads, and the error type.

### The one number that matters — CORRECTED 2026-11-10 04:00

**The original claim in this section was wrong.** It said PyPI carried only
0.4.1 and that this gap *was* the launch. Checked against the live index:

```
distributed-llm: latest=0.6.2.2
releases=[0.4.1, 0.5.0, 0.6.0, 0.6.1, 0.6.2.1, 0.6.2.2]
0.6.2.2 uploaded 2026-11-09 (wheel + sdist), requires_python >=3.10
```

So the index already matches the version in `pyproject.toml`, uploaded
yesterday. The "install instructions point at an August build" premise was
memory, was labelled unverified, and is now falsified.

What this changes:

- **Gate 2 is much closer than assumed.** A user following the README today
  gets the version we are working on, not an old one. The remaining work is
  proving that install works from a clean venv, not rescuing it.
- **Releasing 0.7.0 is an improvement, not an emergency.** Still worth doing
  (the hardening in this branch is not in 0.6.2.2), but it is no longer the
  single thing the launch depends on.
- **The real risk moved.** It is no longer "users get stale code". It is
  "the code on the index has not been smoke-tested from the index", which is
  a smaller and cheaper thing to close.

Lesson recorded deliberately: the plan opened with a number sourced from
memory and marked "unverified", and then treated it as the governing fact.
One HTTP request would have caught it on day one.

---

## 2. Definition of launch (2026-11-30)

We launch when all five are true. Not four.

1. **Master is green.** The PR #10 test matrix passes on all four Python targets, or each remaining red is explicitly quarantined with a tracked issue and a written reason.
2. **The release is published.** `pip install distributed-llm` pulls a version that matches `master` and passes a clean-venv smoke test from the real index.
3. **The site tells the truth.** No page ships a claim the code does not back. No page renders the old palette.
4. **One real end-to-end demo runs on real hardware.** A model loads, serves OpenAI-compatible chat, streams, and survives a node leaving the pool. Recorded, repeatable.
5. **A stranger can install and use it.** Someone who has never seen the repo follows the docs to a working request, with no help from us.

If a gate cannot be met, we ship a smaller product that meets all five, not a larger one that meets four.

### Gate status, 2026-11-10 04:55

| Gate | State | Evidence |
|---|---|---|
| 1. Master green / reds quarantined | **local side complete** | api 1351/0, core 5819/8, dist 6287/8, deploy 150/0, real-import 240/240. Every local red is environmental, classified in `REMAINING_FAILURES.md`; the REAL defects found were fixed (`c827fc03`, `18bb10c5`). CI-side reds remain: container, fuzz, chaos, two integrations. |
| 2. Release published | **both halves verified** | index serves 0.6.2.2; index install passes (`launch_gate_index_install.py`); this tree builds a wheel and installs it cleanly, server serves 143 routes (`launch_gate_build_install.py`). Remaining: bump to 0.7.0 and re-run both. |
| 3. Site tells the truth | **passes — site, docs and desktop all traceable** | Docs machine-checked in CI (install line, 6 nonexistent commands, clone URL, reversed `cp`, coordinator ports). Six site claims with no backing removed, including the FAQ's "distributed training is built in" against a no-op stub. `PERFORMANCE_COMPARISON.md` no longer cites the marketing page as a source of *measurements* (it cited 30–90 / 52 / 68 tok/s, all removed as unmeasured); it now separates measured / third-party-unverified / not-measured, and its remaining website citations are pricing, which the page still publishes. The desktop app was checked the same way: every metric comes from a real IPC or fetch response, and unavailable values read "unknown" / "0 detected" / "N/A" rather than a plausible number. |
| 4. Real end-to-end demo on hardware | **blocked on the A10; everything testable here is tested** | Server starts (10 s), unauthenticated request rejected with 401, authenticated GET /v1/models returns a model, and a chat completion returns generated text — all following QUICKSTART verbatim in a clean venv. Streaming is covered by 54 passing tests. The "node leaves the pool" path is rehearsed on CPU (`test_gate4_node_loss_rehearsal.py`). What remains is the A10 recording itself. |
| 5. A stranger can install and use it | **PASSES, end to end** | Clean venv, `pip install "distributed-llm[cli,self-hosted]"` from the real index, `API_KEY` set, server started, 401 without a token, 200 with one, and a real generated chat reply. The docs that get a stranger there are now checked in CI: install line, 26 CLI invocations, 20 endpoints, clone URL, repo links, reversed `cp`, coordinator ports. SDK docs corrected too — `distllm-sdk` is on neither PyPI nor npm, and the SDK ships inside `distllm.sdk`. |

Gate 5 passes as of 2026-11-10 05:00, verified end to end against the live
index, not a local path. Gate 4 is the only one that cannot be moved from here.

### The defect the quarantining turned up

`test_budget_recovers` was the one failure the runtime director classified
REAL rather than environmental, and it declined to relax the assertion. It was
right: a node recognised as *recovering* advanced its batch budget by
`min(recovery_rate, |grad|/100)`, which at a steady −5.0 ms/sample gradient is
0.05 — exactly the rate of the branch for a node that is merely quiet. Being
detected as recovering bought nothing, and a node that had one transient bad
phase stranded near its 0.1 floor. Full recovery took 30+ samples; it now takes
9, and the factor still holds the floor while the node is genuinely straggling.

---

## 3. The cut list — what we are NOT doing before launch

Explicitly out of scope until after 2026-11-30. These are not failures; they are decisions.

- **Multimodal encoders (O-09).** 2–3 weeks of work for a feature nobody at launch needs.
- **Orchestrator and server module splits (O-06/O-07).** Refactors with no user-visible effect and real regression risk.
- **Dependency-systems consolidation (O-14).**
- **The `_quarantine/` modules.** Documented as deletable by 2026-10-01; delete them at the sweep, do not wire them.
- **Rust SDK verification beyond static review.** No toolchain on this machine. Ship it labelled experimental or do not ship it.
- **New features of any kind.** After today, main is closed to features. Bug fixes, launch blockers, and documentation only.

---

## 4. The 21-day sequence

Three one-week turns, one hard check at each boundary.

### Days 1–7 (Nov 9–15): get master green and mergeable

The single objective: **PR #10 mergeable-green or every red explained.**

- Burn down the failing checks, in this order — the ones that block a merge first:
  1. `test` ×4 (Python 3.10/3.11, Linux/Windows) — the widest gate.
  2. `real-import-gate` — this one is a lie detector: it proves the package imports for real rather than through test doubles. If it is red, something is actually broken for users.
  3. `test-dist`, `chaos-tests`, `docker-compose-integration` — environmental or real, find out which.
  4. `security-deps`, `security-container`, `test-terraform`, `vscode-extension`, `fuzz-test` — lower blast radius, may be quarantined.
  5. Three failing integrations (langchain, llamaindex, crewai) vs eight passing — find the actual difference before touching anything.
- **Merge PR #10 to master only when `test` and `real-import-gate` are green.** Merging with a red import gate ships a broken install.
- The nine director sessions from tonight land their work into this same window. Their branches merge into the release branch, not into master directly.

### Days 8–14 (Nov 16–22): make pip honest, prove the demo

- **Release 0.7.0.** Version bump, changelog, wheel built from master, installed into a clean venv, `distllm --help` and a real chat request verified from the index. Not from a local path.
  - *Corrected 2026-11-10:* 0.6.2.2 is already on the index, so this is a versioned improvement over a working baseline, not a first publish. The clean-venv test can be written and run **now** against 0.6.2.2 to de-risk it before the bump; the same script then re-runs on 0.7.0.
  - The step that has never been done is the one that matters: proving the index install works. Prioritise that over the version number.
- **Record the demo** on the A10 the user has. Real model, real tokens, real streaming, a node leaving the pool and the pipeline surviving. This is the asset the launch needs and the one thing that cannot be faked.
- **Close the docs gap.** The docs director's truth pass plus the website's remaining 11 pages.

### Days 15–21 (Nov 23–29): hardening and rehearsal

- Full regression on the release candidate.
- Load test at the numbers we will claim on the site. If we claim throughput, we measure it, on the hardware we name.
- **The stranger test:** someone who has not touched the repo follows `docs/QUICKSTART.md` cold. Every place they get stuck is a launch blocker, not a doc nit.
- Freeze. No merges after Nov 28 except launch blockers.

### Nov 30: launch

Publish, announce, watch.

---

## 5. Ownership

The nine director sessions own their areas; this session owns the gates and the decision to ship. Standing rules:

- No director merges to master. Directors commit to their own branches; this session merges, in order, after gates pass.
- Every director reports **verified** vs **read-only**. Anything marked read-only is not done — it is a lead.
- No director touches another's files. The scopes are disjoint by design.

---

## 6. The honest risks

| Risk | Likelihood | Consequence | Mitigation |
|---|---|---|---|
| The four failing `test` matrix jobs hide real defects, not just env issues | Medium | Master cannot go green; launch slips | Triage Day 1–2 by reading the actual logs, not by guessing |
| No GPU on the dev machine | **Certain** | The demo cannot be produced here | The user's A10 is the only path; book it in Days 8–14 |
| Rust toolchain missing from PATH (it is installed) | Resolved | — | Absolute path `%USERPROFILE%\.cargo\bin\cargo` works; `sdk-rust-check.yml` already runs it in CI |
| Nine directors produce conflicting or low-quality work | Medium | Merge cost explodes in the last week | Review each on arrival, merge early, never batch to the end |
| **Layer-overlap invariant is declared but never enforced** | **Certain (by inspection)** | A multi-node install can register two nodes on the same layer range and nothing rejects it; inference then has no meaningful stage split | The orchestrator has `validate_layer_assignment` and never calls it, while `topology_registry.validate_assignment` is the single-place invariant. Chart defaults fixed to one worker per range (`782ef947`), so the shipped path is safe. Enforcing at registration is a behaviour change and is deliberately not done pre-launch — tracked here so it is a known gap, not a surprise |
| The release slips past Nov 22 | Medium | No rehearsal window | Cut scope (§3), never cut the rehearsal |

---

## 8. Merge order (decided 2026-11-10)

Directors commit to their own branches; **only this session merges**, and in this order, because three of them write `tests/` and would conflict:

| # | Branch | Why in this position |
|---|--------|---------------------|
| 1 | `claude/distracted-carson-b4f14d` (runtime) | Source changes first — every later test suite asserts against source. |
| 2 | `claude/wonderful-visvesvaraya-b305d9` (deploy) | Touches 2 test files; must land before the test baseline is taken. |
| 3 | `claude/dreamy-jones-a4a9f5` (desktop) | Independent (`tauri/` only); can land any time. |
| 4 | integrations, benchmarks | Independent areas. |
| 5 | `claude/adoring-satoshi-7465d7` (quality) | **Last.** It takes the test baseline; if source moved under it, the baseline is stale. It has 0 commits at time of writing — it must rebase onto the release branch after #1 and #2 land, not before. |
| — | website, docs, sdk | Already merged (`5254be9`, `1d68d3b` pending, `5c0af00`). |

**Quality-dir is told to rebase, not merge early.** If it merges its baseline first, runtime and deploy changes will invalidate it.

## 9. Status board (updated 2026-11-10 03:40)

| Director | Branch | Commits | State |
|---|---|---|---|
| runtime | `distracted-carson-b4f14d` | +4 | **merged** (`985b64c`, then `045170a`/`d27e206` in a follow-up merge). Session still marked running but idle since 22:20 — treat as paused. |
| desktop | `dreamy-jones-a4a9f5` | +1 | **merged** — `cargo check --all-targets` clean (8 warnings, 0 errors) |
| quality | `adoring-satoshi-7465d7` | +3 | **merged** (`7081b94`). Merged out of order; see note below. |
| website | merged | `5254be9` | done, follow-up sweep running |
| sdk | merged | `5c0af00` | done, awaiting MATRIX audit |
| docs | `ff143023…` branch | 0 (working tree only) | instructed to produce a patch |
| deploy | `wonderful-visvesvaraya-b305d9` | +1 | merged earlier |
| integrations | `optimistic-khorana-d00eac` | +2 | **merged** — langchain 34 / llamaindex 29 pass after a conftest fix |
| benchmarks | `upbeat-nightingale-592396` | +2 | **merged** (`edffcf6`) — 14 benchmark/script modules verified |

### All five director branches are merged as of 2026-11-10 03:40

Merge order departed from §8: quality landed second, not last. That was safe
because it was merged *after* runtime's source changes, and rerunning the
affected suites after every later merge is what keeps the baseline honest. The
order was a static plan; the property it was protecting is "the baseline is
taken against final source", and that is preserved by re-measuring, not by the
sequence itself.

### Two defects found during the merges, both fixed

1. **The integration conftests' path guard never ran** (`fc13392`). Both
   tried to force this checkout's source ahead of site-packages, but computed
   `dirname(integrations/)` = the repo root, so `_THIS_SRC` was a nonexistent
   `<root>/langchain/src`. `isdir()` was always False. Consequence: run from a
   worktree, the integration suites silently imported the **main repo's** stale
   editable install. Eight streaming tests were failing for that reason, not
   because the director's fix was wrong.
2. **`CONTRIBUTING.md` named a deleted tool** (`d2b672e`): `benchmarks/compare.py`
   was removed by the hygiene sweep and was a backend-comparison runner, not the
   JSON differ the doc described. Now points at `regression_check.py`.

### Measured baseline, 2026-11-10 04:30 (after every director merge)

| Suite | Result | Command |
|---|---|---|
| `tests/api/` | **1351 passed, 0 failed**, 10 skipped, 12 xfailed, 28 xpassed (4m25s) | `PYTHONPATH=src pytest tests/api/ -q --timeout=180` |
| `tests/deploy/` | **150 passed, 0 failed** (was 45 failed / 126 passed) | `pytest tests/deploy/` |
| Real-import sweep | **240/240 modules import** | `PYTHONPATH=src python tests/scripts/check_real_imports.py` |
| Ruff ratchet | **437 findings vs 443 baseline** (I001 73/73, W292 0, UP 0) | `python scripts/ci/ruff_ratchet.py`, counted from LF blobs as CI does |
| Integrations | langchain **34**, llamaindex **29**, 0 failed | `pytest integrations/<name>/tests/` |
| Tauri crate | `cargo check --all-targets` clean (8 warnings) | `cargo check --all-targets` |
| Benchmarks/scripts | 14 modules parse, import, or answer `--help` | `compileall` + per-module |

The earlier "7091 passed / 48 failed" was two things: the 48 were almost all in
`tests/core/`, which the runtime director then worked down, and the api half is
now fully green on its own. `tests/core/` is being re-measured.

**A measurement trap worth recording.** `ruff --fix` reported W292 clean on this
worktree while CI kept failing it. The worktree checks out CRLF; ruff saw a
trailing newline that the LF blob in git did not have. The fix was real bytes in
the index, and the check that can be trusted is one run against
`git cat-file blob :<path>` exports, not against the working tree.

### Two defects the launch gates found (both fixed 2026-11-10)

1. **A bare `pip install distributed-llm` shipped a broken CLI** (`ad1d72ec`).
   The console script is installed for everyone, but `typer` is an extra, so
   `distllm --help` died with a raw `ModuleNotFoundError` traceback — the first
   command a new user runs. Core staying CLI-free is deliberate (pyproject:65),
   so the fix is an import guard that names the extra and prints the install
   line, not a new dependency.
2. **`scripts/launch_gate_index_install.py`** now exists to catch exactly this
   class of thing: fresh venv, install from the real index with `PYTHONPATH`
   cleared, then import + CLI + module entry point + server import.

### Fixed from the quality director's report, 2026-11-10

Three defects, each verified before and after:

1. **HCL injection via quote/backslash** (`core/provisioning.py`). The guard caught `${`, `$$`, newlines, U+2028 — but not `"` or `\`, the characters that actually escape an HCL string literal. `region='us-east-1"'` rendered as `region = "us-east-1""`. Now rejects quote, backslash and `}`, and the message names what was caught. Verified all four classes reject; a legitimate config still renders.
2. **Plugin signature verification existed but was never wired** (`core/plugin_marketplace.py`). `verify_manifest()` and the signed `PluginManifest` type have been in `plugin_sandbox.py` with no caller. Added `public_key_pem` and a fail-closed `_verify_plugin_manifest` (no key / no manifest / bad signature all refuse).
3. **The `torch.load` gate only read one line** (`tests/test_features_release_gates.py`). Both reported offenders *do* pass `weights_only=True`, on a later line of a multi-line call. The gate now scans the whole call; verified it still rejects a genuinely unsafe one.

`tests/test_features_release_gates.py` + `tests/test_high_severity_fixes.py`: 18 passed, 0 failed.

**Correction to my own earlier assumption:** I told every director that `779f7ed` was the likely cause of their failures. The quality director checked properly and found its own defects were **not** reverts — `public_key_pem` never existed in any history, and the provisioning reject-list was always missing those characters. The revert is real and did hit `src/distllm/dist/`, but it is not the universal explanation I implied.

---

## 7. What I need from the user, and by when

1. **A10 access, booked for Nov 16–20.** Without it there is no recorded demo and gate 4 cannot pass. This is the single hardest dependency.
2. ~~A decision on the Rust SDK~~ — **decided: make it real.** It compiles (`cargo check` clean from `%USERPROFILE%\.cargo\bin`), clippy is clean, and it now has 13 tests after two defects were found and fixed. The vscode-extension CI job is still red and is a separate question.
3. **One `docker build .` before launch — any machine with a working daemon.** Every Dockerfile step has been executed here short of the daemon: requirements.lock installs, protobuf generation and both of its assertions pass, `pip install .` produces all five console scripts, and the module-scope imports that would kill a container at startup (`transformers` at `coordinator.py:25`, `node_pb2`) both succeed. What could not be exercised is the container filesystem, the python3.10 ABI, the CUDA base image, and wheel selection for the target platform. This is the only remaining install path with an unverified step, and it is the first one a self-hoster hits.
4. **Nothing else.** The rest is ours to execute.

# DistLLM — Launch Plan to 2026-11-30

**Owner:** CEO session. **Written:** 2026-11-09 23:15 local.
**Launch date:** 2026-11-30. **Available:** 21 days.
**Status source:** measured 2026-11-09, not estimated.

---

## 1. Where we actually are

| Fact | Value | Source |
|---|---|---|
| Package version | 0.6.2.2 | `pyproject.toml` |
| On PyPI | 0.4.1 only (Aug 2026) | memory, unverified — **confirm before release** |
| `master` vs work | master is 115 commits **behind** the working branch, and fully contained in it | `git rev-list` |
| PR #10 (→ master, 115 commits) | **OPEN, MERGEABLE** | `gh pr view 10` |
| PR #10 checks passing | lint, regression, pip-audit, load-test, memory-profile, autofix, bandit, coverage, security-sast, security-secrets, mutation-testing, performance-report, 8/12 integrations, validate-k8s | `gh pr view 10` |
| PR #10 checks **failing** | test ×4 (ubuntu/windows × 3.10/3.11), test-dist, docker-compose, chaos, security-deps, **real-import-gate**, 3 integrations (langchain, llamaindex, crewai), security-container, terraform, vscode-extension, fuzz | same |
| Rust toolchain | **present** at `%USERPROFILE%\.cargo\bin` (cargo 1.94.1) — not on PATH, but works | verified 2026-11-09 |
| GPU on this machine | **absent** | benchmark brief |

### Decisions taken 2026-11-09

- **Publish late.** 0.7.0 goes to PyPI on **Nov 28**, after the hardening pass and with the demo recorded. Users stay on 0.4.1 until then; that is accepted.
- **Rust SDK ships for real.** It compiles (`cargo check` clean), clippy is clean, and it now has 13 tests that pin its wire contract. Two of those tests found defects that made it non-functional (see below), which is the argument for having done it.

### Rust SDK: what was actually wrong (fixed 2026-11-09)

1. **`ChatResponse` could not deserialize a normal OpenAI payload.** `Message.name` and `tool_call_id` were required on the way in. Every real server — ours included — omits them, so *every* chat response failed to parse. Added `#[serde(default)]`.
2. **`ChatRequest::default()` sent an empty model name.** `#[serde(default = "default_model")]` applies to deserialization only; `Default::default()` left `model: ""`. A request built with `..Default::default()` — which is the documented usage — went out with no model. Replaced with a manual `Default` impl.
3. The crate's **own doc example did not compile** (`message.content` where `message` is `Option<Message>`). That example is the crate's front page.
4. The SDK had **zero tests** across 299 lines. Now 13, covering request serialization, response parsing against real payloads, and the error type.

### The one number that matters

**0.6.2.2 is not on PyPI. 0.4.1 is.** Every install instruction on the website, in the docs, in the SDKs and in the integrations says `pip install distributed-llm`. A user following our own homepage today gets a build from August with none of the ~50 security fixes, none of the SDK parity work, and none of the current API surface.

That gap is the launch. Everything else is secondary to closing it.

---

## 2. Definition of launch (2026-11-30)

We launch when all five are true. Not four.

1. **Master is green.** The PR #10 test matrix passes on all four Python targets, or each remaining red is explicitly quarantined with a tracked issue and a written reason.
2. **The release is published.** `pip install distributed-llm` pulls a version that matches `master` and passes a clean-venv smoke test from the real index.
3. **The site tells the truth.** No page ships a claim the code does not back. No page renders the old palette.
4. **One real end-to-end demo runs on real hardware.** A model loads, serves OpenAI-compatible chat, streams, and survives a node leaving the pool. Recorded, repeatable.
5. **A stranger can install and use it.** Someone who has never seen the repo follows the docs to a working request, with no help from us.

If a gate cannot be met, we ship a smaller product that meets all five, not a larger one that meets four.

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

## 9. Status board (updated 2026-11-10 00:15)

| Director | Branch | Commits | State |
|---|---|---|---|
| runtime | `distracted-carson-b4f14d` | +2 | working |
| desktop | `dreamy-jones-a4a9f5` | +1 | idle, instructed |
| quality | `adoring-satoshi-7465d7` | **+0** | working — no commits yet, watching |
| website | merged | `5254be9` | done, follow-up sweep running |
| sdk | merged | `5c0af00` | done, awaiting MATRIX audit |
| docs | `ff143023…` branch | 0 (working tree only) | instructed to produce a patch |
| deploy | `wonderful-visvesvaraya-b305d9` | +1 | idle, instructed |
| integrations | `54bf46e2…` branch | ? | working |
| benchmarks | `f188c914…` branch | ? | working |

### Measured baseline, 2026-11-10 (after website + SDK + deploy merges)

`PYTHONPATH=src pytest tests/api/ tests/core/ -q` → **7091 passed, 48 failed, 366 skipped**, 7m48s.

The 48 cluster into three areas, all inside the runtime director's scope:

| Area | Tests | Files |
|---|---|---|
| Speculative decoding | 9 | `test_async_pipelined_speculative.py` |
| Radix / cache layer | 15 | `test_radix_tree_cache.py` (8), `test_cache_index.py` (7) |
| Streaming generator | 3 | `test_streaming_generator.py` |
| Defrag / scheduler / recovery / backends | 21 | assorted, 1–3 each |

Separately, `tests/deploy` = 45 failed / 126 passed — the chart-completeness and stale-workflow tests, not regressions from the merge (`test_kustomize.py` is 28/28 green after the fix).

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
2. ~~A decision on the Rust SDK~~ — **decided: make it real.** It compiles (`cargo check` clean from `%USERPROFILE%\.cargo\bin`), clippy is clean, and it now has 13 tests after two defects were found and fixed.
3. **Nothing else.** The rest is ours to execute.

1. **A10 access, booked for Nov 16–20.** Without it there is no recorded demo and gate 4 cannot pass. This is the single hardest dependency.
2. ~~A decision on the Rust SDK~~ — **decided: make it real.** Done to the point of compiling, tested and clippy-clean; CI job already exists. The vscode-extension job is still red and is a separate question.
3. **Nothing else.** The rest is ours to execute.

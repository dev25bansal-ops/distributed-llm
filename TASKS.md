# TASKS.md

> Priority-ordered task list for Claude Code. Derived from the DistLLM Strategic Analysis (June 2026).
> Work top-to-bottom. Do NOT skip ahead. Each task has acceptance criteria.

## 🔴 Critical — Do These First (Weeks 1-3)

### TASK-001: Fix SEC-01 — Remove pickle.load (RCE risk)
- **Find:** `rg 'pickle\.load' src/`
- **Fix:** Replace with JSON, msgpack, or safetensors. If pickle is genuinely required, use `pickle.RestrictedUnpickler` with a strict class allowlist.
- **Test:** Add `tests/security/test_no_pickle.py` that fails if any `pickle.load` is added to `src/`.
- **Done when:** 0 `pickle.load` calls in `src/`, regression test merged.

### TASK-002: Fix SEC-02 — Remove eval()/exec() calls (33 instances)
- **Find:** `rg '\beval\(|\bexec\(' src/`
- **Fix:** Replace each with `ast.literal_eval` (for literals), `json.loads` (for JSON), `ast.parse` + manual walk (for expressions), or `importlib` (for plugin loading).
- **Test:** Add ruff rule `S307` (eval) and `S102` (exec) to `pyproject.toml` to fail CI on any new usage.
- **Done when:** 0 `eval`/`exec` in `src/` (or each surviving one has a documented justification + `# noqa` with reason).

### TASK-003: Fix SEC-03 — Remove verify_ssl=False (4 instances)
- **Find:** `rg 'verify_ssl.*=.*False|verify=False' src/`
- **Fix:** Remove the flags. If for development, gate behind `DISTLLM_INSECURE=1` env var that logs a WARNING on every use.
- **Done when:** 0 `verify_ssl=False` in `src/`. Any dev-only escape hatch is env-gated and logged.

### TASK-004: Fix SEC-04 — Audit .secrets.baseline
- **Run:** `detect-secrets audit .secrets.baseline`
- **Fix:** Manually verify each entry. Rotate any real secrets found (GitHub, HuggingFace, cloud provider).
- **Test:** Verify CI job `scripts/ci/check_secrets_baseline.py` blocks PRs on baseline drift.
- **Done when:** Baseline audited, real secrets rotated, CI enforcement verified.

### TASK-005: Fix SEC-05 — Audit subprocess calls (56 instances)
- **Find:** `rg 'subprocess\.(run|call|Popen|check_output)' src/`
- **Fix:** For each call, trace input sources. Add Pydantic validation for any user-influenced argument. Use `shutil.which` for path resolution.
- **Doc:** Document each surviving subprocess call in `docs/SECURITY_HARDENING.md` with rationale.
- **Done when:** All 56 calls audited, user-influenced inputs validated, doc updated.

---

## 🟡 High — Code Quality Cleanup (Weeks 4-7)

### TASK-006: Add ruff rules to prevent regression
- **Edit:** `pyproject.toml` → `[tool.ruff.lint]` → add rules: `T20` (flake8-print), `BLE001` (blind-except), `G004` (logging-f-string), `S307` (eval), `S102` (exec).
- **Run:** `ruff check src/ --statistics` to see current violations.
- **Done when:** New rules added, existing violations listed as separate tasks below.

### TASK-007: Replace print() with loguru (1,429 instances)
- **Find:** `rg 'print\(' src/`
- **Strategy:** Script this. Use a Python script (not manual) to convert `print("X", var)` → `logger.info("X", var=var)`. Skip `tests/`, `examples/`, `scripts/` (exempt via per-file ignores).
- **Don't:** Convert all 1,429 in one PR. Do in batches of 50-100 per PR to keep reviews manageable.
- **Done when:** 0 `print()` in `src/distllm/` (excluding exempted files).

### TASK-008: Replace bare `except Exception:` (191 instances)
- **Find:** `rg 'except Exception:' src/`
- **Strategy:** For each, decide: (a) catch specific exception type, (b) re-raise as domain exception from `distllm.errors.types`, or (c) if genuinely catch-all, log with full traceback before swallowing.
- **Done when:** 0 `except Exception:` in `src/distllm/` (excluding tests).

### TASK-009: Replace time.sleep in async contexts (154 instances)
- **Find:** `rg 'time\.sleep' src/`
- **Strategy:** For each, check if function is `async def`. If yes, replace `time.sleep(N)` with `await asyncio.sleep(N)`. If no, leave as-is but document.
- **Done when:** 0 `time.sleep` in `async def` functions in `src/distllm/`.

### TASK-010: Replace f-string logger calls (691 instances)
- **Find:** `rg 'logger\.[a-z]+\(f' src/`
- **Strategy:** Script this. Convert `logger.info(f"x={var}")` → `logger.info("x={var}", var=var)`. Mechanical refactor.
- **Done when:** 0 f-string logger calls in `src/distllm/`.

---

## 🟢 Strategic — Product & Traction (Weeks 8-12)

### TASK-011: One-command install script (R-01)
- **Create:** `install.sh` at repo root — `curl -sSL https://distllm.ai/install | bash`
- **Behavior:** Detect OS, install Python if missing, create venv, `pip install distllm[self-hosted]`, download TinyStories-1M, print "Run: distllm chat".
- **Test:** Test on macOS, Ubuntu, Windows (WSL). Document at `docs/QUICKSTART.md`.
- **Done when:** Script works on 3 OSes, doc updated.

### TASK-012: HuggingFace Spaces demo (R-07)
- **Create:** `huggingface-spaces/Dockerfile` + `app.py` — runs DistLLM with TinyStories-1M, exposes Gradio chat UI.
- **Deploy:** Push to HF Spaces. Get public URL.
- **Add:** "Try it now" button on `website/index.html` linking to the Space.
- **Done when:** Live HF Space URL, website link added, README references it.

### TASK-013: TCO calculator on website (R-22)
- **Edit:** `website/js/calculator.js` (exists) — build full TCO comparison: self-hosted (DistLLM on owned GPUs) vs cloud (Together AI per-token pricing).
- **Inputs:** Current monthly cloud spend, available GPUs, expected growth rate.
- **Output:** 3-year TCO comparison, breakeven month, savings.
- **Add:** Lead-gen form (email to see results) at `website/tco-calculator.html`.
- **Done when:** Calculator live, lead form saving to `/tmp/leads.jsonl` (will wire to CRM later).

### TASK-014: RBAC permission matrix documentation (R-12)
- **Edit:** `docs/SECURITY_HARDENING.md` — add section "RBAC Permission Matrix".
- **Format:** Table with rows = actions (inference, list models, admin users, view audit log, etc.), columns = 6 roles (admin, model-admin, auditor, inference-only, read-only, viewer).
- **Add:** `distllm rbac describe` CLI command in `src/distllm/cli/` that prints the matrix.
- **Done when:** Matrix documented, CLI command works, tests added.

### TASK-015: Benchmark blog post content (R-27)
- **Run:** `make bench` with TinyStories-1M, GPT-2, Llama-3.2-7B (if GPU available). Compare single-node vLLM vs DistLLM 2-node vs DistLLM 4-node.
- **Create:** `website/blog/distllm-vs-vllm-benchmarks.html` with results table + charts.
- **Include:** Honest results. If DistLLM loses on single-node, say so. Win on multi-node.
- **Done when:** Blog post published, raw benchmark JSON in `benchmarks/results/`.

---

## 🔵 Background — Architectural Decisions (Ongoing)

### TASK-016: Concurrency model decision (CQ-04)
- **Document:** New ADR at `docs/adr/0006-concurrency-model.md` — pick async-first OR thread-first. Not both.
- **Recommendation:** Async-first (FastAPI is async, gRPC is async-friendly, I/O-bound workload). Migrate `threading.Thread` → `asyncio.Task`, `threading.Lock` → `asyncio.Lock`, `threading.Event` → `asyncio.Event`.
- **Migration plan:** Module by module. Start with `coordinator.py` (highest impact).
- **Done when:** ADR accepted, migration plan documented, first module migrated.

### TASK-017: Backend tiering (ARCH-04)
- **Document:** `docs/BACKEND_PLUGIN_GUIDE.md` — add "Backend Tier Policy" section.
- **Tier 1 (full support):** vLLM, llama.cpp. Documented, tested, security-tracked.
- **Tier 2 (community):** PyTorch. Best-effort.
- **Tier 3 (deprecated):** TensorRT, ExLlamaV2, ONNX. Mark for removal in v0.6.0 unless community maintains.
- **Code:** Add `@backend_tier("tier1")` decorator in `src/distllm/backends/registry.py`.
- **Done when:** Tier policy documented, decorators added, `distllm backends list` shows tier.

### TASK-018: Reorganize `src/distllm/core/` (ARCH-02)
- **Plan:** Move 130+ flat files into subdirectories:
  - `core/cache/` — all cache_*.py files
  - `core/speculative/` — speculative_*.py files
  - `core/cost/` — cost_*.py files
  - `core/federation/` — federation-related files
- **Migration:** Use `rope` or `pyrefly` for automated refactor. Update all imports.
- **Done when:** `core/` has < 30 top-level files, all imports updated, tests pass.

---

## 📋 How to Use This File

### For Claude Code sessions:
1. Read this file at session start
2. Ask user: "Which task should I work on?"
3. Pick the highest-priority incomplete task
4. Follow the "Done when" criteria as acceptance definition
5. Update this file: change `- [ ]` to `- [x]` when complete

### For the founder:
- Work top-to-bottom. Don't skip ahead.
- Each TASK is one PR (or one batch of PRs for TASK-007, -008, -010).
- If a TASK takes more than 1 week, split it.
- Add new TASKS as they emerge from the strategic analysis.

### Priority colors:
- 🔴 Critical = blocks YC application, blocks production deploy
- 🟡 High = blocks code quality bar, blocks enterprise pilots
- 🟢 Strategic = drives traction, needed for YC application
- 🔵 Background = architectural, do incrementally over months

### Task status format (add `[x]` when done):
- [ ] TASK-001: Not started
- [~] TASK-001: In progress (add PR link)
- [x] TASK-001: Complete (add PR link + date)

---

## Progress Tracker

### Critical (Weeks 1-3)
- [ ] TASK-001: SEC-01 pickle.load
- [ ] TASK-002: SEC-02 eval/exec
- [ ] TASK-003: SEC-03 verify_ssl=False
- [ ] TASK-004: SEC-04 secrets baseline
- [ ] TASK-005: SEC-05 subprocess audit

### High (Weeks 4-7)
- [ ] TASK-006: Add ruff rules
- [ ] TASK-007: print() → loguru
- [ ] TASK-008: bare except cleanup
- [ ] TASK-009: time.sleep in async
- [ ] TASK-010: f-string logger

### Strategic (Weeks 8-12)
- [ ] TASK-011: One-command install
- [ ] TASK-012: HuggingFace Spaces demo
- [ ] TASK-013: TCO calculator
- [ ] TASK-014: RBAC matrix docs
- [ ] TASK-015: Benchmark blog post

### Background (Ongoing)
- [ ] TASK-016: Concurrency model ADR
- [ ] TASK-017: Backend tiering
- [ ] TASK-018: Reorganize core/

---

**Source:** DistLLM Strategic Analysis Report (June 2026), Sections 2, 3, 10, 15.
**Update cadence:** Weekly review. Reorder priorities as needed.

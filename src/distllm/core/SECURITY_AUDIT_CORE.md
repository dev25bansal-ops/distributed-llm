# Security Vulnerability Catalog — `src/distllm/core/`

**Scope:** All `.py` files under `src/distllm/core/` (357 source files, ~54.5k LOC scanned).
**Method:** Static read of each candidate file + `bandit 1.9.4` over the package, cross-checked with targeted grep for `torch.load`, `pickle`, `subprocess`, `urllib`, `eval/exec`, `shell=True`, `verify=`, SQL builders, and secret literals.
**Date:** 2026-07-13.

---

## TL;DR

The core package is **substantially hardened** — the historically dangerous classes are mostly *already mitigated* by design:

- ✅ **Every `torch.load(...)` uses `weights_only=True`** (7/7 sites) → no unsafe deserialization RCE via checkpoints.
- ✅ **No `eval`/`exec`/`os.system`/`os.popen`/`shell=True`** anywhere on tainted input.
- ✅ **No hardcoded API keys / tokens / passwords** in core.
- ✅ **`DISTLLM_NO_AUTH` / `DISABLE_AUTH` are rejected** (auth is now always-on).
- ✅ **SQL** uses parameterized queries with identifier-only interpolation (no user-value injection).

**Real findings (sorted by severity):** 1 Medium (SSRF header-trust bypass), 1 Medium (plugin integrity fail-open), 2 Medium (insecure temp dirs), 1 Medium (SSRF urlopen with no TLS/SSRF guard), a Low/path-dependent `trust_remote_code` allowlist-bypass, a Low internal path-traversal, and several Low informational items. **No Critical or High issues were confirmed.**

---

## Findings (sorted by severity)

### 1. [MEDIUM] SSRF / IP-spoofing via `X-Forwarded-For` trust in tests — CWE-348 / CWE-918
**CVSS 3.1 (base): 6.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N)** — lower if proxy is never enabled.
**File:line:** `src/distllm/api/ip_utils.py:25-26` (also `:15-27`, `:48-56`)

```python
def _is_trust_proxy_enabled() -> bool:
    value = os.environ.get("DISTLLM_TRUST_PROXY_HEADERS", "")
    if value.lower() in ("1", "true"):
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):   # <-- always True under pytest
        return True                       # <-- implicit trust during tests
    return False

def get_client_ip(request, *, trust_proxy=None):
    if trust_proxy is None:
        trust_proxy = _is_trust_proxy_enabled()
    if trust_proxy:
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
        forwarded = request.headers.get("X-Forwarded-For", "")
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[0]          # <-- attacker-controlled, never validated
    return request.client.host if request.client else "unknown"
```

**Exploit:** `PYTEST_CURRENT_TEST` is set by pytest, so *any* run with tests imported (or a server started from a test session, a common pattern) implicitly trusts `X-Forwarded-For` / `X-Real-IP`. An attacker sends `X-Forwarded-For: 1.2.3.4` and any IP-based control that calls `get_client_ip` (rate-limit, audit IP, ban) resolves to the spoofed value. This defeats the auth rate-limit (`middleware.py:223,233`) and poisons audit logs (`request_auditor.py`).

**Expected:** proxy headers only honored when an operator explicitly sets `DISTLLM_TRUST_PROXY_HEADERS=1`.
**Actual:** auto-trusted whenever pytest is on the path; `X-Forwarded-For` value is unvalidated (no IP-format/range check).
**Fix:** Drop the `PYTEST_CURRENT_TEST` implicit-trust branch (set `DISTLLM_TRUST_PROXY_HEADERS=1` in test fixtures instead); validate/parse `X-Forwarded-For` entries through `ipaddress.ip_address` and reject private/loopback before trusting.

---

### 2. [MEDIUM] Plugin integrity check fails OPEN (default) — CWE-345 / CWE-494
**CVSS 3.1 (base): 6.3 (AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H)** — requires attacker to place a file in a *trusted* plugin dir, but the control is otherwise silently inert.
**File:line:** `src/distllm/core/plugin_system.py:386,389-398` (and the same logic in `plugin_marketplace.py` via `_PLUGIN_NAME_RE`).

```python
strict = self._config.get("verify_plugins", True)   # default True
allowlist_path = path.parent / "distllm_plugin_hashes.txt"
if not allowlist_path.exists():
    if strict:
        return False                                    # fail-closed ONLY if strict
    logger.warning(...)                               # otherwise:
    return True                                        # <-- loads UNVERIFIED plugin
```

**Exploit:** The `strict` flag is read from `self._config.get("verify_plugins", True)`. The `install_plugin`/discovery path does not populate `verify_plugins` from a safe default — if the operator never sets `verify_plugins: true` in the config object actually passed in, the allowlist is missing → the plugin is loaded **without integrity verification** while only logging a warning. A malicious `.py` dropped into a trusted plugin dir executes `exec_module` on import (`plugin_system.py:469`). The docstring says "fail-open by default" — contradicting the constant-time security intent.

**Expected:** missing allowlist ⇒ reject (fail-closed) unless operator explicitly opts into dev mode.
**Actual:** fail-open when the config key is absent or falsy.
**Fix:** Default `strict = self._config.get("verify_plugins", False)` → invert: default to **fail-closed**; require an explicit opt-out. Same pattern must be applied to `plugin_marketplace._verify_plugin_manifest` (currently requires a public key, which is good — keep that).

---

### 3. [MEDIUM] SSRF in cross-cluster cache migration — `urllib.request.urlopen` with no SSRF/TLS guard — CWE-918 / CWE-20
**CVSS 3.1 (base): 5.8 (AV:N/AC:H/PR:N/UI:N/S:U/C:L/I:L/A:L)** — needs reachable migration endpoint.
**File:line:** `src/distllm/core/cache_migration.py:97-104` and `:122-129`

```python
req = urllib.request.Request(
    f"{cluster_url.rstrip('/')}/api/v1/cache/warm",
    data=payload, headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req) as resp:    # <-- no scheme/host validation, no TLS verify control
    if resp.status != 200:
        all_success = False
```

`cluster_url` is operator-supplied in normal use, but the same `CacheMigrator` is reachable from cluster-to-cluster sync; if `source_url`/`dest_url` is ever influenced by a less-trusted control plane it becomes a blind SSRF primitive (can hit `http://169.254.169.254/...` or internal services, and `file://` is permitted by `urlopen`). No `SSLConext` is supplied, so TLS verification relies on global defaults and cannot be enforced.

**Fix:** Validate scheme ∈ {http,https}, resolve & reject private/loopback/link-local hosts (reuse `webhook_manager._is_safe_webhook_url` logic), and pass an explicit `context=ssl.create_default_context()` with verification on. Note `webhook_manager.py` already implements a solid SSRF guard (`_is_safe_webhook_url`, `:41-96`) — reuse it so the two code paths stay consistent.

---

### 4. [MEDIUM] Insecure hardcoded `/tmp` directories — CWE-377 (insecure temp file/dir)
**CVSS 3.1 (base): 5.5 (AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:N)** — local attacker can pre-create symlinks / plant files.
**File:line:**
- `src/distllm/core/adaptive_compression.py:128` — `output_dir = os.path.join(self._output_base or "/tmp/distllm-compress", f"{os.path.basename(model_path)}-{tag}")`
- `src/distllm/core/adaptive_compression.py:225` — `output_dir: str = "/tmp/distllm-compress"`
- `src/distllm/core/distributed_distillation.py:62` — `checkpoint_dir: str = "/tmp/distllm-distillation"`

`os.makedirs(output_dir, exist_ok=True)` then writes model checkpoints / compressed weights into a **fixed, world-predictable `/tmp` path**. A local attacker can pre-create `/tmp/distllm-compress` as a symlink to a privileged location, or a similarly-named file, achieving symlink-follow / data tampering before write (TOCTOU). `basename(model_path)` is derived from a HuggingFace ID or path supplied to `compress()`.

**Fix:** Use `tempfile.mkdtemp(prefix=..., dir=<configured, per-user dir>)` and/or enforce `os.chmod(0o700)` on creation; reject `model_path` containing path separators / `..`.

---

### 5. [LOW / path-dependent] `trust_remote_code` allowlist-bypass via crafted model name — CWE-829 / CWE-494
**CVSS 3.1 (base): 4.2 (AV:N/AC:H/PR:H/UI:N/S:U/C:L/I:L/A:L)** — only relevant when `trust_remote_code` is left at default and an operator loads an attacker-named model.
**File:line:** `src/distllm/models/partitioner.py:70-104` (note: *not* in `core/`, but it is the trust-decision function used by `core/coordinator.py`, `inference_engine.py`, `activation_profiler.py`, `cluster_manager.py`, `adaptive_compression.py`).

```python
TRUSTED_MODELS_ALLOWLIST = _TRUSTED_FROM_REGISTRY | {"baichuan","baichuan2","chatglm",...}
def _should_trust_remote_code(model_name, trust_remote_code=None):
    if trust_remote_code is not None:
        return trust_remote_code
    model_lower = model_name.lower().split("/")[-1]
    family = model_lower.split("-")[0].split(".")[0]
    for trusted in TRUSTED_MODELS_ALLOWLIST:
        if model_lower == trusted or family == trusted:
            return True
    return False
```

The family match (`family == trusted`) means any HF repo whose *last path segment* starts with, e.g., `chatglm` (e.g. `evilorg/chatglm-pwn` → family `chatglm`) flips `trust_remote_code=True`, causing `from_pretrained(..., trust_remote_code=True)` to **execute arbitrary Python** shipped in the model repo's `modeling_*.py`. This is a real RCE primitive if an attacker can influence `model_name` (e.g. a multi-tenant endpoint where `model` is taken from the request body — see `api/cost_middleware.py:72`, `api/streaming.py:308`).

**Expected:** only an explicit operator opt-in enables remote code.
**Actual:** a *substring-prefix family match* against the HF repo name auto-enables it. The comment at `:96-100` acknowledges the bypass risk for `my-qwen-exploit` but misses that *trusted* family names are themselves exploitable.
**Fix:** Match against a **full, exact model ID** (e.g. `org/model`) from a curated list, never a derived prefix; or require `trust_remote_code=True` to be set only via explicit config and refuse name-based auto-trust.

---

### 6. [LOW] Internal path traversal in KV-cache persistence (defense-in-depth) — CWE-22
**CVSS 3.1 (base): 3.1 (AV:N/AC:H/PR:L/UI:N/S:U/C:L/I:N/A:N)** — only if `request_id`/`model_name` become attacker-controlled.
**File:line:** `src/distllm/core/cache_persistence.py:35-36,43,54`

```python
def _get_path(self, request_id: str, model_name: str) -> Path:
    return self._storage_path / model_name / f"{request_id}.pt"
```

`model_name` and `request_id` are concatenated into a filesystem path with **no sanitization**. Today `request_id` is a `uuid4()` (`coordinator.py:1061`) and `model_name` is a server config value, so this is **not currently reachable** from external input. If the persistence API is ever exposed with client-supplied IDs (the batch route at `api/routes/batch.py:36` accepts a *client-assigned* `request_id`), a value like `../../etc/cron.d/x` could write outside `storage_path`.

**Fix:** `model_name`/`request_id` must be validated against `^[A-Za-z0-9_.-]+$` (or hashed) before path construction; use `path.resolve()` + prefix check against `storage_path`.

---

### 7. [LOW] `plugin_sandbox.run_sandboxed` — capability model is *partially* enforced (informational)
**File:line:** `src/distllm/core/plugin_sandbox.py:223-224, 226-232, 237`

```python
if not policy.allows(_SUBPROCESS_CAP) and _looks_like_subprocess_escape(cmd):
    raise PermissionError("SUBPROCESS capability required ...")
env = dict(os.environ)
if not policy.allows(PluginCapability.ENV_READ):
    for secret in ("API_KEY","ANTHROPIC_API_KEY","OPENAI_API_KEY","DATABASE_URL","AWS_SECRET_ACCESS_KEY"):
        env.pop(secret, None)
if not policy.allows(_NETWORK_CAP):
    env["DISTLLM_SANDBOX_NO_NET"] = "1"
...
env = _scrub_env(policy)        # re-does the same scrub (correct, idempotent)
```

**Assessment — the model IS mostly enforced, with gaps:**
- ✅ Shell metacharacters (`; & | $ \``) and shells (`sh/bash/cmd/powershell`) are rejected outright (`:220-221`).
- ✅ `SUBPROCESS` capability gates external binaries (`:223`).
- ✅ Secret env vars are scrubbed when `ENV_READ` is not granted.
- ⚠️ **`DISTLLM_SANDBOX_NO_NET=1` is a *convention flag only*** — it does **not** actually block network egress. There is no `unshare(CLONE_NEWNET)` / `seccomp` / macOS `sandbox-exec` call; the comment at `:16-19` admits this and defers to a future WASM path. So a plugin without `NETWORK` can still make outbound connections unless the host OS enforces it. **The capability is advisory, not a hard isolation boundary.**
- ⚠️ Secret scrubbing covers only 5 fixed var names; a plugin granted `ENV_READ` sees *all* env (including other secret keys not in the list).
- ⚠️ Filesystem capability (`FILESYSTEM_WRITE`) is checked only for `cwd` (`:234`), not for the command's own file operations; the sandbox never restricts *which* paths a granted plugin may touch beyond `allowed_paths` for cwd.

**Recommendation:** Treat the sandbox as *defense-in-depth*, not a trust boundary. For untrusted plugins, require the (currently `NotImplementedError`) `run_wasm` path or an OS-level net namespace.

---

## Looked-like-a-vuln-but-SAFE (verified)

| # | Location | Why it is safe |
|---|-----------|------------------|
| S1 | `torch.load` ×7 — `cache_persistence.py:67`, `cache_snapshot.py:111`, `kv_backup.py:147`, `kv_cache.py:1107`, `bargaining_engine.py:247`, `distributed_distillation.py:198`, `cache_persistence.py` | **All use `weights_only=True`** → no arbitrary-code deserialization. `bargaining_engine.py:245-247` even has a comment citing the exact CWE. |
| S2 | `plugin_marketplace.install` / `plugin_system.install_plugin` (`plugin_marketplace.py:237-240`, `plugin_system.py:530-533`) | `subprocess.run([sys.executable,"-m","pip","install", package])` is **list-form, no shell**; `package` is `f"distllm-plugin-{plugin_name}"` after `_PLUGIN_NAME_RE` / `_sanitize_plugin_name` (alphanumeric/dash/underscore only) and a **fail-closed** `_plugin_install_allowed` gate (`plugin_system.py:43-58`: remote install disabled unless `allow_remote_plugin_install` or `trusted_plugins`). |
| S3 | `subprocess.run` in `gpu_power_manager.py`, `device_registry.py`, `autonomous_healer.py`, `provisioning.py` | All **list-form with fixed binaries** (`nvidia-smi`, `sysctl`, `terraform`). No tainted args, no `shell=True`. |
| S4 | `pickle.dumps` in `advanced_scheduling/disaggregated.py:224-225` | Serializes *internal* KV tensor objects for in-cluster transfer; **no `pickle.loads` of untrusted data** is reachable in `core/` (the only `pickle.loads` is in `dist/dist/zero_copy.py:67`, which round-trips torch CUDA IPC storage handles produced locally — not attacker input). |
| S5 | `pgvector_store.py` SQL (`_table()` at `:56-59`) | Table/collection name is sanitized: `safe = "".join(ch for ch in self.collection_name if ch.isalnum() or ch=="_")`, then wrapped in double quotes → **identifier injection prevented**; all values use `%s` placeholders. |
| S6 | `evaluation_harness.py:280`, `prompt_library.py:289` SQL | `clause`/`where` are built only from **hardcoded column names** (`model_id = ?`, `dataset = ?`, `tags_json LIKE ?`) with user values passed as bound params. The f-string interpolation inserts only the *structure*, never user text → **no SQLi**. |
| S7 | `DISTLLM_NO_AUTH` / `DISABLE_AUTH` (`api/middleware.py:204-213`) | Both are now **explicitly rejected** with `logger.critical`; auth is always required. The old dev-mode bypass was removed. |
| S8 | Auth: `api_key_store.py:117-127` | Token compared via `hashlib.sha256` + `hmac.compare_digest` (constant-time); raw keys are never logged (`get_display_key` warns against logging). |
| S9 | `secret_manager.py` (Vault/AWS/file backends) | FileSecretBackend `chmod 0o600`; Vault/AWS use `https://`/SDK with no `verify=False`. No TLS verification is disabled. |
| S10 | `certificate_manager.py` | `rsa` key generated 2048-bit (`:327`), SHA-256 signatures, private key `chmod 0o600` (`:362`), and `_encryption_algorithm` prefers `BestAvailableEncryption` when a passphrase is set (falls back to `NoEncryption` *with a warning*). No weak crypto. |
| S11 | `webhook_manager.py:41-96` `_is_safe_webhook_url` | Solid SSRF guard — blocks non-http(s), private/loopback/link-local/multicast/reserved, `localhost`/`.local`/`.internal`, and resolves hostnames to reject internal IPs. This is the **correct** pattern `cache_migration.py` should mirror. |
| S12 | Plugin symlink / trusted-dir checks (`plugin_system.py:428-457`) | Correctly resolves symlinks and uses `Path.relative_to` + `startswith(os.sep)` to avoid `/trusted_evil` prefix bypass. |
| S13 | `request_auditor.py` / `security/log_redaction.py` | PII redaction utilities exist and `redact()` is wired into moderation (`api/routes/moderation.py:409`) and the log-redaction wrapper. Secrets/PII are redacted in logs (API keys via `sk-...{20,}` pattern, SSN, card, IP). **Caveat:** redaction is *regex-based* and only applied where callers explicitly invoke `redact(...)`; it is not a global log sink, so any `logger.info(f"...{raw}...")` that bypasses the helper can still leak. |

---

## Cross-cutting assessments

**Are secrets redacted in logs?** *Partially.* A PII/secret regex redactor exists (`request_auditor.PiiInspector.redact`, `security/log_redaction.LogRedactor`) and is used in moderation and the error wrapper. However, it is **opt-in per call-site** — there is no global logging filter that scrubs every emitted record. Raw prompts, model paths, and any accidental `logger` calls with sensitive interpolations are **not** automatically redacted. Recommend a `logging`/`loguru` sink-level filter applying `LogRedactor.redact` to *all* records.

**Is the sandbox capability model enforced?** *Partial (see #7).* The subprocess sandbox correctly denies shells, enforces `SUBPROCESS`/`ENV_READ` capabilities, and scrubs a fixed set of secret env vars. But `NETWORK` denial is a **no-op flag** (`DISTLLM_SANDBOX_NO_NET`), and the actual isolation depends on the host OS. Treat as defense-in-depth, not a hard boundary.

**SQL injection in query builders?** *Not present.* All user values are parameterized; only fixed identifiers are interpolated, and the one dynamic identifier (`pgvector_store._table`) is allowlist-sanitized.

**Hardcoded secrets / `verify=False` / weak crypto?** *None found* in `core/`. No `md5`/`sha1` for security, no ECB, no hardcoded keys, no `verify=False`.

---

## Prioritized fix list
1. **Plugin integrity fail-open** (`plugin_system.py:386`) — default to fail-**closed**.
2. **`X-Forwarded-For` implicit trust under pytest + unvalidated header** (`api/ip_utils.py:25-26,48-56`) — remove pytest auto-trust, validate parsed IPs.
3. **SSRF in `cache_migration.py`** (`:97-104,122-129`) — reuse `webhook_manager` SSRF guard + enforce TLS verify.
4. **Insecure `/tmp` model dirs** (`adaptive_compression.py:128,225`, `distributed_distillation.py:62`) — `tempfile.mkdtemp` + `0o700`.
5. **`trust_remote_code` family-prefix bypass** (`models/partitioner.py:96-104`) — exact full-ID match only.
6. **Harden `plugin_sandbox`** — make `NETWORK` denial real (net namespace) or document it as non-isolating; scrub all secret-named env vars, not 5 fixed names.
7. **Global log redaction sink** — apply `LogRedactor` to every log record, not per-call.
8. **Validate `request_id`/`model_name`** in `cache_persistence._get_path` (defense-in-depth).

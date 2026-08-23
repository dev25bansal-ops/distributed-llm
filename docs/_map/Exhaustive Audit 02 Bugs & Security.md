---
tags:
  - audit
  - exhaustive
date: 2026-08-11
---

# Exhaustive Audit 02 — Bugs & Security (Medium/Low)

**← [[Exhaustive Audit 2026-08-11]]**

All findings in category `bug, security` (Medium/Low and non-verified severities).

**87 findings** — Medium: 75 · Low: 12

---

### F-060 — [Medium] Proto layout is inconsistent across three locations; `make proto` writes to an unused dir and one module imports a missing package

`Makefile:61` · zone=`tooling-tests` · category=`bug`

- **Summary:** Committed+imported protobuf stubs live at `src/distllm/dist/node_pb2[._grpc].py` (matching mypy override `distllm.dist.node_pb2`), but `make proto` regenerates into `src/distllm/communication/` - a directory that is not where any code imports from - so `make proto` would create stale duplicates. Independently, `src/distllm/core/distributed_speculative.py` (lines 241, 586) lazily imports `from distllm.proto import node_pb2[,_grpc]`, but `src/distllm/proto/` does not exist; that path will raise ModuleNotFoundError at runtime the first time remote-draft gRPC stub is built. proto/node.proto exists, so a valid regeneration is possible.
- **Evidence (verbatim):**
```
make proto: --python_out=src/distllm/communication/ ... but committed stubs are src/distllm/dist/node_pb2.py and distributed_speculative.py does `from distllm.proto import node_pb2_grpc`
```
- **Impact:** Regeneration via the documented Makefile produces a non-functional layout; the remote-draft gRPC feature and its CI test (tests/core/test_distributed_speculative.py, run in ci.yml) can fail at runtime on a genuine code path.
- **Effort:** 2-4 hours
- **Reliability:** Confirmed: no src/distllm/proto dir; stubs only under dist/; Makefile targets communication/.
- **Recommendation:** Point `make proto` at `--python_out=src/distllm/dist/` (matching where stubs are committed), add a CI step that regenerates and fails on `git diff`, and fix distributed_speculative.py to import `distllm.dist.node_pb2[,_grpc]` (or create src/distllm/proto and keep it consistent).

---

### F-061 — [Medium] One-api auto-registration sends no admin auth and the spurious fork recomputes JSON for embeddings; FastAPI middleware forwards stale Content-Length

`integrations/one-api/src/distllm_one_api/__init__.py:98` · zone=`integrations` · category=`bug`

- **Summary:** distllm_one_api apply() POSTs to `{admin_url}/api/channel` with no Authorization/one-api token header, so registration fails 401 on a real one-api instance and it silently falls back to printing config. Separately, distllm_fastapi middleware.py forwards every inbound header (only popping host) while also re-sending the raw body; a stale `Content-Length` (and forwarded `Authorization`/`accept-encoding`) conflicts with httpx's own encoded body and can cause 400s from the upstream; it also forwards the original full path when prefix differs from /v1.
- **Evidence (verbatim):**
```
resp = httpx.post(                     f"{admin_url}/api/channel",                     json=config,                     timeout=10.0,                 )
```
- **Impact:** one-api auto-registration is effectively non-functional in real deployments, and the FastAPI proxy can emit malformed requests to the upstream when headers disagree with the re-encoded body.
- **Effort:** 2-3 hours
- **Recommendation:** Read one-api admin token from env/param and pass it (e.g. `Authorization: Bearer <token>` or the oneapi `oneapi` header). In middleware.dispatch, filter hop-by-hop headers (Content-Length, Transfer-Encoding, Connection) before forwarding and always resolve the upstream target by prefix rather than echoing request.path.

---

### F-062 — [Medium] Spark _call_batch_udf retry guard is dead code that always re-raises, and the pandas_udf closure captures an unpicklable async client

`integrations/spark_connector.py:508` · zone=`integrations` · category=`bug`

- **Summary:** In `_call_batch_udf` (spark_connector.py:508-515) the except block checks `if any(attempt < self._max_retries for attempt in range(self._max_retries))` which is always True (attempt 0 < max_retries), so it unconditionally re-raises; a single transient failure on one row aborts the entire Spark executor UDF. The comment 'Re-raise so the outer retry loop handles it' is wrong because _infer_udf has no outer retry loop. Separately, transform()/transform_stream() register a pandas_udf whose lambda closes over the whole DistLLMSparkTransformer (including DistLLMClient + per-executor state), which is not a safe pattern for Spark serialization.
- **Evidence (verbatim):**
```
except Exception:                     if any(                         attempt < self._max_retries                         for attempt in range(self._max_retries)                     ):                         # Re-raise so the outer retry loop handles it                         raise
```
- **Impact:** Spark batch/stream transform is not resilient to transient inference errors and risks executor serialization failures; the apparent retry logic provides zero protection.
- **Effort:** 3-5 hours
- **Reliability:** any(attempt<max_retries for attempt in range(max_retries)) is vacuously True; _infer_udf (line 455) has no internal retry, so one failed prompt raises through the pandas_udf and drops the whole partition.
- **Recommendation:** Replace the dead guard with a real per-row retry loop (mirroring _process_one_batch's max_retries+backoff) that records an empty string on final failure instead of raising; refactor the UDF to build the client inside the executor rather than closing over self.__client.

---

### F-063 — [Medium] Version drift: pyproject 0.4.1 vs src __version__ 0.4.0 vs stale published wheel 0.4.0

`pyproject.toml:7` · zone=`tooling-tests` · category=`bug`

- **Summary:** pyproject.toml declares 0.4.1 but `distllm.__version__` reports 0.4.0 and the only built wheel/sdist in dist/ are 0.4.0 dated Jun 2 - nearly all current src modules (736 non-__init__ .py files) postdate that wheel, so the published artifact is missing most of the package. No CI job rebuilds/verifies the sdist+wheel against src, so packaging regressions (missing modules, broken entry points) go undetected until release.
- **Evidence (verbatim):**
```
version = "0.4.1" (pyproject) vs __version__ = "0.4.0" (src/distllm/__init__.py:7) vs built wheel dist/distributed_llm-0.4.0-*.whl (built Jun 2)
```
- **Impact:** The Python-package version the runtime reports (0.4.0) disagrees with the metadata (0.4.1); anyone installing the published wheel gets a partial build. SECURITY.md claims 0.4.x is the supported line, compounding trust issues.
- **Effort:** 2-4 hours
- **Reliability:** Verified via stat timestamps and version greps - no CI references a package-build job.
- **Recommendation:** Single-source the version (e.g. `__version__` read from a version file / `importlib.metadata` in __init__), add a CI job that `python -m build` the sdist+wheel on PRs and asserts (1) every src/distllm/*.py is present in the wheel, (2) `import distllm` from a clean venv importing the wheel works, (3) the 4 console scripts resolve. Rebuild+republish 0.4.1.

---

### F-064 — [Medium] 12 CI steps and Makefile install a non-existent `testing` extra

`pyproject.toml:82` · zone=`tooling-tests` · category=`bug`

- **Summary:** CI jobs install extra `testing` which is not defined in pyproject.toml. pip emits 'WARNING: distributed-llm does not provide the extra testing' and installs only `dev`. Any testing-only dependency that was intended (or that dev doesn't carry) is silently absent, which can explain environment-specific collection/runtime failures (e.g. missing psutil on some runners if a venv is reused).
- **Evidence (verbatim):**
```
piper: 12 workflow steps do `pip install -e ".[dev,testing]"` (ci.yml:48/74/113/177, gpu-tests.yml x4, benchmark-regression.yml, compression-ci.yml...) but [project.optional-dependencies] defines no `testing`
```
- **Impact:** Non-hermetic installs: CI envs are not reproducible from declared metadata; the discrepancy between the extra name referenced and declared is a silent failure factory.
- **Effort:** 2-4 hours
- **Reliability:** grep for 'testing' shows 12 references; grep pyproject shows no 'testing' extra.
- **Recommendation:** Either add a `testing` extra to `[project.optional-dependencies]` (move pytest, pytest-asyncio, hypothesis, pytest-benchmark there and keep `dev` including `testing`) or replace every `.[dev,testing]` with `.[dev]` in the four workflows. Add a CI lint that asserts every referenced extra exists in pyproject.

---

### F-065 — [Medium] middleware.py logs the full generated API key to the logger at every startup, contradicting its own fingerprint-only comment

`src/distllm/api/middleware.py:165` · zone=`api-gateway` · category=`security`

- **Summary:** In _get_or_generate_api_key (middleware.py 139-171), when API_KEY is unset the server generates a random 48-byte key and, despite the comment at 157 explicitly stating 'Log a fingerprint, not the full key. Full keys in logs = credential leak', the code emits the entire secret via logger.warning at line 165: f"Generated API key: {generated_key} ". This runs at module import (line 176) on every process start, so the full credential is written to application logs, which are typically captured by aggregators/CI and stored at rest — the exact leak the neighboring comment warns against. SHA-256 of a random token is fine for a fingerprint; logging the plaintext key is not.
- **Evidence (verbatim):**
```
logger.warning(f"API_KEY not set. Generated a secure random API key.\n...\nGenerated API key:\n{generated_key}\n") (161-169); full key is emitted inside the warning (165)
```
- **Impact:** Generated admin credential persisted in cleartext in logs/aggregators; if logs leak, full API access is disclosed. Violates the module's own stated security intent. CVSS ~4.4.
- **Effort:** 1-2 hours
- **Reliability:** Trigger: start the server with API_KEY unset. The module-level call `_get_or_generate_api_key()` (line 176) runs; the warning string interpolates `generated_key` verbatim. Inspect the process log/stdout — the full 48-byte key is present in cleartext.
- **Recommendation:** Only log the fingerprint (the existing `fingerprint` variable) and require the operator to set API_KEY; if a generated dev key must be surfaced, print it once to stdout with an explicit 'do not ship these logs' marker rather than through logger, and add a log-redaction rule for the generated value.

---

### F-066 — [Medium] Latent arbitrary file write via upload filename (absolute path / ../) in routes/files.py; router unmountable because require_coordinator is undefined

`src/distllm/api/routes/files.py:135` · zone=`api-gateway` · category=`security`

- **Summary:** routes/files.py upload_file builds the output path as `file_path = upload_dir / (file.filename or "unnamed")` (line 135) with no sanitization of the client-supplied filename. In pathlib, `Path(a) / "/abs"` yields the absolute path `/abs`, and `Path(a) / "../x"` escapes the file-id directory; upload uses write_bytes (138), so an attacker controlling the multipart filename can write arbitrary content to any absolute path the server process may write (e.g. overwriting application/config files), up to the 512MB cap. Separately, the router cannot be mounted: line 29 imports `require_coordinator` from ..auth_deps, but auth_deps.py only defines require_role — require_coordinator is referenced by 5 route files (files, api_keys, experiments, fine_tuning, webhooks) and defined nowhere in src. Both issues are currently latent because files_router is NOT included in server.py, but the file-write is one mount away from a Critical.
- **Evidence (verbatim):**
```
file_path = upload_dir / (file.filename or "unnamed") (135); file_path.write_bytes(content) (138); from ..auth_deps import require_coordinator (29)
```
- **Impact:** Server-side arbitrary file write (Code Execution potential) if the route is ever mounted; broken/unusable route definitions across 5 modules. CVSS ~8.8 if reachable.
- **Effort:** 3-5 hours
- **Reliability:** Upload a multipart field with filename="/etc/cron.d/x" or "..%2F..%2Fconfig" to POST /v1/files (once imported). file_path = <uploads>/<file_id>/ + absolute-or-relative filename; write_bytes places attacker content at that resolved path. The require_coordinator import error means the module currently cannot even be imported (ImportError) — it is dead until the dependency is fixed, and then immediately creates an arbitrary-file-write surface.
- **Recommendation:** Sanitize filename with os.path.basename() and reject names containing '/' , '\', '..', or leading '.'; never join an absolute client path. Either define require_coordinator in auth_deps (and enforce it as a real role check) or remove the broken import before mounting. Do not mount files_router until both are fixed.

---

### F-067 — [Medium] OIDC/OAuth2 SSO state & nonce stores grow unboundedly with no TTL reaper; GenericOAuth2Handler skips CSRF state check on empty state

`src/distllm/api/sso_auth.py:441` · zone=`api-gateway` · category=`security`

- **Summary:** In sso_auth.py, OIDCHandler._state_store and _nonce_store and GenericOAuth2Handler._state_store grow one entry per get_login_url() call, and entries are removed ONLY when a successful handle_callback pops the exact state (sso_auth.py 265, 315, 455). There is no periodic reaper; if any login-URL-generating endpoint becomes reachable, an attacker can drive unbounded dict growth (memory DoS). Separately, GenericOAuth2Handler.handle_callback gates the state check behind `if expected_state:` (454) — an empty state bypasses CSRF protection at the handler layer (it only happens to be enforced because the /v1/auth/token route independently requires state, sso_middleware.py 551-561).
- **Evidence (verbatim):**
```
self._state_store[state] = time.time() + self._state_ttl (441); popped only on successful callback (455); OIDC nonce_store updated per login (231-238) with no reaper
```
- **Impact:** Memory-exhaustion DoS on a reachable login flow; weakened CSRF enforcement at the handler layer if the route-level check is ever refactored.
- **Effort:** 2-3 hours
- **Reliability:** Rebind get_login_url to an HTTP route (or call it in a loop in-process) → each call inserts an entry into _state_store/_nonce_store forever unless the exact state is later presented to handle_callback. Entries with time.time()+600 expiry are never swept. For the state-bypass: GenericOAuth2Handler.handle_callback('code', expected_state='') skips the whole CSRF block (454 `if expected_state:`) and proceeds to exchange the code.
- **Recommendation:** Add a periodic prune of expired entries to _state_store/_nonce_store (as already done for _revoked_tokens/_cleanup_revoked, lines 604-611); replace the 'if expected_state' guard with a fail-closed 'state is required' check that returns None when state is absent.

---

### F-068 — [Medium] Base PagedAttentionManager.get_kv_cache concatenates full blocks without slicing by num_tokens, returning zero-padded tail garbage

`src/distllm/backends/paged_attention.py:427` · zone=`dist-partition` · category=`bug`

- **Summary:** get_kv_cache does torch.cat(keys, dim=2) over full KVCacheBlock.key_cache tensors (each block size max_tokens). For a sequence whose last block is partially filled (num_tokens < max_tokens), the concatenated tensor includes unused zero slots, so the caller sees block_size*num_blocks tokens instead of the real num_tokens. The quantized variant (QuantizedPagedAttentionManager.gather_kv) correctly slices :block.num_tokens — the base manager does not.
- **Evidence (verbatim):**
```
if block.key_cache is not None:     keys.append(block.key_cache)   # FULL block, no [:block.num_tokens] return torch.cat(keys, dim=2), torch.cat(values, dim=2)
```
- **Impact:** Attention over KV reads reads zero-padded invalid tokens, corrupting generation for any sequence with a partially-filled final block and inflating memory reads.
- **Effort:** 1-2 hours
- **Reliability:** Trigger: allocate_sequence(seq, 20) with block_size 16 → two blocks; last block.num_tokens becomes 4 only after append_token writes there; get_kv_cache returns 32-slot tensors. Compare quantized gather_kv line 275-277 slicing [:block.num_tokens,:].
- **Recommendation:** Slice [:block.num_tokens,:] per block before cat, mirroring gather_kv; add a test with a non-multiple-of-block_size token count.

---

### F-069 — [Medium] QuantizedPagedAttentionManager INT4 stores INT8-width (no packing) but reports 0.25x compress_ratio — KV memory savings overstated 2x

`src/distllm/backends/paged_attention_quantized.py:149` · zone=`dist-partition` · category=`bug`

- **Summary:** For quant_method='int4', _quantize clamps to [-7,7] and stores as torch.int8, and _allocate_block_storage allocates shape (2,heads,block_size,head_dim) int8. The block never packs 4-bit values (docstring claims head_dim//2 uint8 packing). compress_ratio is reported 0.25 while actual storage is 0.5 (int8/2 bytes vs fp16/2 bytes = 0.5). Real KV memory footprint is twice what the manager reports.
- **Evidence (verbatim):**
```
quantized = (tensor / scale).clamp(-7, 7).to(torch.int8)  # int8, NOT uint8 packed ... self._quant_dtype = torch.int8 ... "compress_ratio": 0.5 if quant_method == "fp8" else 0.25
```
- **Impact:** Oversold 2x KV memory savings can cause the scheduler to over-commit KV capacity and OOM or evict under pressure.
- **Effort:** 1-2 days
- **Reliability:** Repro: instantiate with quant_method='int4'; block.key_quantized.shape is (2,heads,16,head_dim) int8 (lines 165-173); _quantize line 148-150 clamps to int8. compress_ratio line 124 still claims 0.25.
- **Recommendation:** Either actually pack 4-bit (store 2 values/byte, head_dim//2) with proper unpack in _dequantize, or set compress_ratio=0.5 for int4 and document that INT4 is int8-width emulation until packing is implemented.

---

### F-070 — [Medium] WebGPU backend reports is_available()==True yet every forward raises NotImplementedError — advertises unusable backend

`src/distllm/backends/webgpu_backend.py:200` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** WebGPUNodeAdapter.is_available() returns True whenever aiortc is importable, but _forward_via_workers raises NotImplementedError, so a registered WebGPU backend appears in list_available_backends() yet cannot serve any request. The registry's health probe (object.__new__ without __init__) masks this by failing closed on health_check for un-instantiated adapters, but direct construction + load_model() + forward() is a hard failure path.
- **Evidence (verbatim):**
```
raise NotImplementedError(     "WebGPU worker dispatch is a placeholder — refusing to return zero logits...")
```
- **Impact:** A 'production' deployment that auto-selects WebGPU from the available list fails immediately at inference time instead of degrading to a working backend.
- **Effort:** 2-3 hours
- **Reliability:** is_available returns True on aiortc presence (lines 244-248); forward() lines 182-186 route to _forward_via_workers which raises.
- **Recommendation:** Return False from is_available() until real WebRTC dispatch is wired, and/or gate forward() behind a clearly named 'enabled' flag with a one-way property. Add a registry conformance test asserting that any backend listed as available can actually complete a forward.

---

### F-071 — [Medium] Spot orchestrator silently substitutes fabricated static market listings when a provider API fails

`src/distllm/cloud/spot_orchestrator.py:964` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** _SaladProvider._list_fallback returns hardcoded static GPU entries and prices whenever the real Salad containers API raises, and list_instances returns those fabricated listings as normal. SpotOrchestrator.find_cheapest then returns them and launch_cluster places real bids (max_price=inst.price_per_hour) at fabricated prices. A provider outage or transient error can therefore cause real money to be bid on invented inventory.
- **Evidence (verbatim):**
```
def _list_fallback(...): known_gpus: list[dict] = [{"gpu_type": "NVIDIA RTX 4090", ... "price": 0.45}, ...]
```
- **Impact:** Real expenditure on nonexistent GPU inventory during provider hiccups; worsens the spot-orchestrator 'find cheapest then book' risk thread.
- **Effort:** 2-3 hours
- **Reliability:** list_instances (line 917-962) catches Exception and returns self._list_fallback(...); find_cheapest (1372-1392) and launch_cluster (1411-1427) consume the result and bid.
- **Recommendation:** On provider API failure, return [] (empty) and let the orchestrator report 'provider unreachable' rather than fabricating inventory. If static data must be surfaced, tag instances with an explicit 'stale/estimated' flag and refuse to bid on them.

---

### F-072 — [Medium] Settings loaded via from_profile / ProfileConfig do not apply DISTLLM__* env-var precedence that the docs promise

`src/distllm/config/profiles.py:10` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** DistLLMSettings.from_yaml carefully builds env precedence via EnvSettingsSource(cls)() and deep-merges YAML beneath it (settings.py:377-381). But DistLLMSettings.from_profile -> cls.model_validate(merged) does not feed env values, and ProfileConfig.load (profiles.py) only merges YAML+profile+preset — its docstring line 10 claims 'Environment variables (DISTLLM__*) -- always take final precedence' yet load() never reads env. Two load entry points therefore give different env-handling behavior, and any DISTLLM__* override can be silently ignored in profile mode.
- **Evidence (verbatim):**
```
4. Environment variables (DISTLLM__*) -- always take final precedence
```
- **Impact:** Operators setting secrets/tuning via env vars get different results depending on whether they load YAML or YAML+profile, risking misconfiguration and env-secret override loss.
- **Effort:** 2-3 hours
- **Reliability:** settings.py from_profile (392-405) uses model_validate directly without EnvSettingsSource, unlike from_yaml (377-383).
- **Recommendation:** Route from_profile through the same env-aware merge as from_yaml (call DistLLMSettings.from_yaml with the resolved YAML path+profile dict, or inject EnvSettingsSource results into the merged dict before model_validate). Add a test asserting a DISTLLM__* var overrides both YAML and profile/preset.

---

### F-073 — [Medium] AuditTrail (SQLite/JSONL) is billed as 'immutable' but stores no tamper-evidence (no hash chain/MAC)

`src/distllm/core/aegis_compliance.py:252` · zone=`core-priv-sec` · category=`security`

- **Summary:** `AuditTrail.record` appends entries to SQLite (WAL) or a JSONL buffer with no integrity protection: no chained hashes, no HMAC, and `entry_id=uuid4().hex[:16]` is random (not a sequence, so nothing detects a missing/gapped entry). Any process with filesystem or DB access can insert, delete, or rewrite rows and there is no detection mechanism. For SOC 2 / HIPAA 'immutable audit' claims this is a gap.
- **Evidence (verbatim):**
```
entry_id = uuid.uuid4().hex[:16] ...self._conn.execute('INSERT INTO audit_entries (entry_id, event_type, ...) VALUES (...)')
```
- **Impact:** An attacker with storage access can silently alter the compliance audit history, undermining the 'immutable' audit premise.
- **Effort:** 1 day
- **Reliability:** Open aegis_audit.db, UPDATE an audit_entries row (e.g., change result to 'allowed'); export() returns the modified row with no indication of tampering.
- **Recommendation:** Link entries into a hash chain: store `prev_hash` and `hash = HMAC(secret, prev_hash || canonical_entry)` with the key held outside the DB/file; on query/export recompute and verify each link and surface any break. At minimum document that the log is not tamper-evident without a signing key.

---

### F-074 — [Medium] Aegis ModelWatermark uses a hardcoded public salt as the de-facto key; watermark is removable and unverifiable from weights (CWE-321 / CWE-699)

`src/distllm/core/aegis_compliance.py:69` · zone=`core-priv-sec` · category=`security`

- **Summary:** `aegis_compliance.ModelWatermark` embeds message bits in weight LSBs at positions deterministically derived from the PUBLIC constant `_WATERMARK_SALT = b"distllm-aegis-watermark-v2"`. A thief who reads the (open-source) code can recompute `_select_unique_indices` and zero out exactly the marked LSBs. Likewise `extract()` reads the `_aegis_watermark_meta` attribute stored on the module, not the weight signal, so deleting that attribute removes all evidence. The 'imperceptible, ownership-verifying' claim provides no real tamper-resistance.
- **Evidence (verbatim):**
```
_WATERMARK_SALT = b"distllm-aegis-watermark-v2" ...h = hashlib.sha256(_WATERMARK_SALT + str(i).encode()).hexdigest() idx = int(h, 16) % total
```
- **Impact:** Model-theft watermark can be stripped by any attacker with source access; provenance claim is unverifiable.
- **Effort:** Half day
- **Reliability:** Repro: read _WATERMARK_SALT from source, rebuild index list, clear LSB of those params; or `del model._aegis_watermark_meta` -> extract() raises 'No watermark metadata' before any weight check.
- **Recommendation:** Derive index selection and the integrity tag from a per-deployment secret key (HMAC with a stored key) rather than a hardcoded salt; tie extraction to the weight values (e.g., recompute the expected LSB pattern at deterministic positions) instead of a removable module attribute; and verify the SHA-256 tag before returning the message. Prefer the keyed HMAC path in security/watermark.py, which already takes a secret_key.

---

### F-075 — [Medium] API key authentication runs PBKDF2 (100k iters) per stored key per request before any rate-limit short-circuit — CPU-exhaustion DoS + timing side channel

`src/distllm/core/api_key_store.py:145` · zone=`core-priv-sec` · category=`security`

- **Summary:** `ApiKeyStore.authenticate` re-hashes the presented token with PBKDF2-SHA256 (100,000 iterations) against EVERY stored key's salt in a loop before any limiter short-circuits. The middleware calls `store.authenticate(token)` first; only afterwards does it inspect the rate limiter. With N keys, each request costs ~N * 100k HMAC ops (tens of ms per key), so an unauthenticated remote attacker can exhaust server CPU by sending arbitrary tokens. Also the loop returns on the first match (or exhausts all keys on failure), leaking key position via timing.
- **Evidence (verbatim):**
```
for k in self._keys:     token_hash = self._hash_key(token, k.salt)  # 100k-iteration PBKDF2 per key     if compare_digest(token_hash, k.key): return (k.key_id, k.role)
```
- **Impact:** DoS amplification proportional to key count; minor timing oracle on key identity. Rate limiter reduces but does not eliminate the per-request cost since authenticate runs before it.
- **Effort:** 4-8 hours
- **Reliability:** N keys => N * 100k HMAC-SHA256 per auth attempt; grep confirms `store.authenticate(token)` is called on the hot path in api/middleware.py:283, api/server.py:999/1062.
- **Recommendation:** Add a fast pre-filter (e.g., an HMAC-SHA256 first-stage index or a Bloom filter) before the expensive PBKDF2 loop; enforce rate limiting before `authenticate` on repeated failures; and bound the total keys scanned. Use a fixed-time loop so iteration count does not leak key position.

---

### F-076 — [Medium] AutonomousHealer records a 'perfect-health' heartbeat when a node dies, and defaults to dry_run

`src/distllm/core/coordinator.py:295` · zone=`core-ops-ha` · category=`bug`

- **Summary:** coordinator._on_node_mark_dead builds a default GPUHeartbeat(node_id=...) with all telemetry zeroed (health_score=1.0, predictor risk=0) and feeds it to the healer, which then treats the newly-dead node as perfectly healthy — the opposite of the intent. Combined with the DISTLLM_AUTO_HEAL_DRY_RUN=1 default, the healer can never actually recover a node in production even when it does detect risk.
- **Evidence (verbatim):**
```
from distllm.core.autonomous_healer import GPUHeartbeat hb = GPUHeartbeat(node_id=node_id) self._auto_healer.record_heartbeat(hb)
```
- **Impact:** The self-healing/spot-fleet feature is a no-op in default production: dead nodes are recorded healthy and GPU reset never executes, so predictive drain/recover never triggers from the coordinator node-failure path.
- **Effort:** 4-6 hours
- **Reliability:** GPUHeartbeat defaults give health_score=1.0 and predictor heuristic risk=0.0 (autonomous_healer.py L188-222), so FAILURE_THRESHOLD 0.3 is never exceeded; dry_run=os.environ DISTLLM_AUTO_HEAL_DRY_RUN default '1' (L1045) makes reset_gpu a no-op always.
- **Recommendation:** Instead of a zeroed heartbeat, pass the real failure event (e.g. record_failure / a DRAINING state) into the healer, and make dry_run default False (or exit loudly when set). Add an integration test where on_node_mark_dead leads to a healer state transition.

---

### F-077 — [Medium] CoordinatorFailoverHandler never touches the HA election protocol it claims to use

`src/distllm/core/coordinator_failover.py:167` · zone=`core-ops-ha` · category=`bug`

- **Summary:** coordinator_failover.py docstring and class name promise HA-protocol-aware failover, but _trigger_failover/_check_tcp_alive only do raw socket.create_connection reachability and pick the first accepting TCP peer as the 'new coordinator'. It ignores leader identity, term, and election output, so it can reconnect to a non-leader or a stale node, and it is dead code outside chaos scenarios.
- **Evidence (verbatim):**
```
def _trigger_failover(self) -> None:     for peer_host, peer_port in peers:         if (peer_host, peer_port) == current: continue         if self._check_tcp_alive(peer_host, peer_port): ... discover new coordinator
```
- **Impact:** If ever wired, workers would fail over to an arbitrary TCP-listening peer rather than the elected leader, breaking the quorum; today the module merely misleads as a dead, wrong abstraction.
- **Effort:** 4-6 hours
- **Reliability:** grep shows CoordinatorFailoverHandler instantiated nowhere in prod (only kraken_chaos references the string 'coordinator_failover'). _check_tcp_alive (L156) is socket.create_connection only.
- **Recommendation:** Either delete it, or rewrite _trigger_failover to query RayFaultTolerance.get_leader()/handle_heartbeat_request and only call on_reconnect for the elected leader. Replace TCP liveness with a heartbeat term check.

---

### F-078 — [Medium] differential_privacy.privacy_budget_used under-reports cumulative epsilon (advanced-composition term dropped)

`src/distllm/core/differential_privacy.py:126` · zone=`core-priv-sec` · category=`security`

- **Summary:** `DifferentialPrivacy.privacy_budget_used` returns `epsilon * sqrt(2*num_queries*ln(1.25/delta))` and claims 'advanced composition (Kairouz et al. 2015)'. The published advanced-composition bound is `epsilon' = k*epsilon*(e^epsilon - 1) + epsilon*sqrt(2*k*ln(1/delta))`. The code drops the linear `k*epsilon*(e^epsilon-1)` term entirely and misplaces delta, so for realistic epsilon (>0.1) and k>1 the reported total epsilon is smaller than the true bound — the privacy spend is under-reported, claiming more privacy than is guaranteed.
- **Evidence (verbatim):**
```
composed_epsilon = (self._config.epsilon * math.sqrt(2 * num_queries * math.log(1.25 / self._config.delta)))
```
- **Impact:** Privacy budget exhaustion is under-stated; tenants may receive more cumulative queries than the advertised epsilon allows, weakening the (eps, delta) claim.
- **Effort:** Half day
- **Reliability:** With eps=1.0, k=10: code returns sqrt(2*10*ln(12.5)) ≈ 7.14; true advanced-composition first term alone = 10*1*(e^1-1) ≈ 17.2 — an underestimate.
- **Recommendation:** Use libsdp/opacus-style accounting (RDP summation + conversion) rather than a hand-rolled composition. If keeping advanced composition, implement both terms: `k*eps*(e^eps-1) + eps*sqrt(2*k*ln(1/delta))`. Add tests comparing against Kairouz et al. for a range of k and epsilon.

---

### F-079 — [Medium] Distillation checkpoint saved from a different thread while training thread is mid-backward

`src/distllm/core/distributed_distillation.py:147` · zone=`core-training` · category=`bug`

- **Summary:** stop() calls _save_checkpoint() from the caller thread while the background loop may be inside loss.backward()/optimizer.step(); reading state_dict/optimizer mid-step can save a torn checkpoint. The default checkpoint_dir '/tmp/distllm-distillation' is also a POSIX-only path.
- **Evidence (verbatim):**
```
def stop(self):     self._should_stop.set()     self._save_checkpoint()
```
- **Impact:** Possible torn/corrupt checkpoints on stop; wrong teacher logits CPU-moved, and Windows path misbehavior.
- **Effort:** 2-4 hours
- **Reliability:** Call stop() while teacher forward+backward executes; saved checkpoint may capture mid-step weights.
- **Recommendation:** Save inside the training thread between optimizer steps (it already saves every 50); stop() only sets the event and joins before final save. Use a platform-neutral cache dir instead of a hardcoded /tmp literal.

---

### F-080 — [Medium] _dp_sample clamps logits to unit L2 norm (sensitivity 2) but sigma assumes sensitivity 1 — up to ~4x privacy overstatement

`src/distllm/core/dp_inference.py:990` · zone=`core-priv-sec` · category=`security`

- **Summary:** `_dp_sample` normalizes logits to unit norm (`logits / max(1.0, norm)`) and adds Gaussian noise of scale `sigma`, with the docstring claiming a (eps,delta)-DP guarantee 'per token when logits are L2-clipped'. Two unit-norm logit vectors differ by up to 2 in L2, so the true L2 sensitivity is 2, not the 1 assumed by `DPConfig.sigma = max_grad_norm*sqrt(2 ln(1.25/delta))/eps` (max_grad_norm=1). Noise needed scales with sensitivity^2, so the epsilon actually achieved is up to ~4x worse than claimed.
- **Evidence (verbatim):**
```
clipped = logits / max(1.0, logits.norm(dim=-1, keepdim=True)) noise = torch.randn_like(clipped) * sigma  # sigma assumes sensitivity = max_grad_norm = 1
```
- **Impact:** Privacy guarantee for the one genuine DP sampling path is materially overstated.
- **Effort:** Half day
- **Reliability:** Two one-hot-constant logit vectors normalized to unit norm differ by 2 in L2; Gaussian mechanism sigma for sensitivity s must scale as s, and RDP cost ~ s^2/(2 sigma^2).
- **Recommendation:** Account for the correct sensitivity: for normalized logits set `sigma` using sensitivity 2 (or clip the logit *difference* to a chosen C, sigma=C*sqrt(2 ln(1.25/delta))/eps) and document the exact epsilon achieved. Note also that adding noise directly to hard logits is not the calibrated Gaussian-mechanism guarantee — prefer a proper report-noisy-max / exponential mechanism or per-token sensitivity analysis.

---

### F-081 — [Medium] request_auditor writes unredacted error/metadata to the JSONL audit file and lacks the long_token/AWS-key patterns

`src/distllm/core/request_auditor.py:162` · zone=`core-priv-sec` · category=`security`

- **Summary:** `RequestAuditor._write_log` serializes `asdict(entry)` including the raw `error` and `metadata` fields to the daily JSONL audit file. If an exception string or metadata value contains an API key / secret, it is persisted unredacted. Separately, request_auditor's `PII_PATTERNS` lacks the `long_token` (base64url run) and `aws_key` patterns that log_redaction added, so the ApiKeyStore auto-generated token_urlsafe(48) admin key is flagged by log_redaction but NOT by the auditor's inspector/redaction — an inconsistent redaction surface.
- **Evidence (verbatim):**
```
log_file = self._log_dir / f"audit-{...}.jsonl" with open(log_file, "a") as f: f.write(json.dumps(asdict(entry)) + "\n")
```
- **Impact:** Sensitive credentials can leak into on-disk audit logs; inconsistent PII detection could surface secrets that the log redactor otherwise masks.
- **Effort:** 2-4 hours
- **Reliability:** Call auditor.record(..., error="Timeout reading sk-abcdef...hijklmnop") -> the audit-YYYYMMDD.jsonl line contains the raw token because asdict(entry) includes error verbatim.
- **Recommendation:** Run `LogRedactor.redact_exception` / `redact` over `error` and `metadata` values (and `ip_address`/`user`) before writing the audit line, and unify the PII pattern set into one shared constant imported by both request_auditor and log_redaction so the auto-generated admin key is handled consistently.

---

### F-082 — [Medium] SharedLayerPool.get_shared_tensor returns aliased tensors with no copy-on-write

`src/distllm/core/shared_layer_pool.py:179` · zone=`core-training` · category=`bug`

- **Summary:** get_shared_tensor returns the SAME torch.Tensor to every model sharing a layer; any in-place op (LoRA merge, quantize, fine-tune update) on one model silently mutates all co-sharing models' weights. The pool tracks ref_counts but gives no protection from write-after-share.
- **Evidence (verbatim):**
```
def get_shared_tensor(self, model_name, layer_name):     ...     return shared.tensor  # single shared tensor handed to every model
```
- **Impact:** Weight poisoning across unrelated models on the same node; sharing becomes a correctness hazard once adapters/tuning mutate a shared layer.
- **Effort:** 2-4 hours
- **Reliability:** m1=get_shared_tensor(A,'l'); m2=get_shared_tensor(B,'l'); m1.mul_(2) changes both A and B to the same values.
- **Recommendation:** Expose read-only semantics; on mutation request, clone-on-write (copy the tensor and bump storage). Document that consumers must not mutate the shared tensor.

---

### F-083 — [Medium] Monthly cost-budget enforcement compares against all-time total_cost, never reset per billing period

`src/distllm/core/usage_meter.py:336` · zone=`core-perf-obs` · category=`bug`

- **Summary:** tenant.total_cost accumulates every record forever (no period boundary in record_request), and check_quota compares quota.cost_budget_per_month against that lifetime total. Once any tenant's cumulative spend crosses the monthly budget it stays blocked forever, even after the month rolls over, because TenantUsage.current_billing_period_start/end are set once at first use and never advanced and totals are never pruned. The daily token cap works only because daily_tokens is keyed by date.
- **Evidence (verbatim):**
```
if quota.cost_budget_per_month > 0:     if tenant.total_cost >= quota.cost_budget_per_month:         ... return False, f'monthly budget ... exceeded' # tenant.total_cost += cost  (unbounded, never reset at month boundary)
```
- **Impact:** A tenant that legitimately exhausts one month's budget is permanently denied service in subsequent months (unless overage_allowed), an availability bug for paid/GA deployments and a cost-accounting correctness error for reporting.
- **Effort:** 2-4 hours
- **Reliability:** Set cost_budget_per_month=100; record 100 units in month M; advance to month M+1 and call check_quota -> still blocked because total_cost (lifetime) >= 100. Confirmed by reading record_request lines 274-287 and check_quota 336-340.
- **Recommendation:** Track spend by billing period (e.g. store a monthly_cost dict keyed by 'YYYY-MM' like daily_tokens) and check the current month's bucket, or reset total_cost when now >= current_billing_period_end. Add a test advancing the clock across a month boundary and asserting budget frees up.

---

### F-084 — [Medium] max_concurrent_requests quota is a check-then-act race (violated under concurrency)

`src/distllm/core/usage_meter.py:357` · zone=`core-perf-obs` · category=`bug`

- **Summary:** check_quota reads self._concurrent.get(tenant_id,0) WITHOUT holding self._lock and returns 'ok' if below the cap; enforce_quota() then calls increment_concurrent() under the lock. Between the unlocked read and the locked increment, N concurrent requests all observe the same under-cap value and all proceed, so max_concurrent_requests can be exceeded arbitrarily. The daily/monthly budget reads in check_quota also access self._tenants/self._quotas outside the lock.
- **Evidence (verbatim):**
```
if quota.max_concurrent_requests > 0:     current = self._concurrent.get(tenant_id, 0)   # no lock     if current >= quota.max_concurrent_requests:         return False, ...     # ... allowed # enforce_quota(): allowed,reason=check_quota(...); then increment_concurrent() locks later
```
- **Impact:** Concurrency caps intended to protect a tenant/backend can be breached by simultaneous bursts, enabling resource oversubscription and cost overrun beyond the configured limit — a real quota-enforcement correctness gap.
- **Effort:** 2-4 hours
- **Reliability:** Set max_concurrent_requests=1, fire two threads that call enforce_quota() nearly simultaneously; both see _concurrent==0 and both pass -> concurrent count reaches 2. Confirmed by read ordering (check outside lock, increment inside lock).
- **Recommendation:** Make the concurrent-check-and-increment atomic: move the reservation into the same lock hold (pass increment=True into a locked check, or use a counting-semaphore per tenant). The daily token and monthly cost checks should also read under the lock for consistency with record_request's writer.

---

### F-085 — [Medium] VLLMPipelineEngine and LlamacppPipelineEngine are byte-identical copies; run_pipeline_overlap misnamed and drops batch/seq fields

`src/distllm/dist/backends/vllm.py:124` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** dist/backends/vllm.py and llamacpp.py each define a full PipelineEngine whose run_pipeline (lines 52-107) is character-for-character identical, differing only in the docstring. vLLM additionally exposes run_pipeline_overlap (lines 124-174) that is NOT overlapping (sequential blocking loop with timeout=self._timeout_s) and omits the batch_size=/seq_len= fields that run_pipeline sets (so the protobuf gets defaults 0). This false 'overlap' API and the duplicated class drift across the two backends.
- **Evidence (verbatim):**
```
def run_pipeline_overlap( ... request = node_pb2.ForwardPassRequest(request_id=request_id, use_cache=True, is_first_pass=is_first)
```
- **Impact:** Duplicate 250-line classes to maintain; a public 'overlap' API that does not overlap and sends different request fields than the same call without 'overlap', risking divergent NodeService behavior.
- **Effort:** 3-4 hours
- **Reliability:** Compare vllm.py:52-107 to llamacpp.py:52-107 (identical); vllm.py:124-174 omits the batch_size/seq_len kwargs present at vllm.py:70-76.
- **Recommendation:** Collapse to a single PipelineEngine parameterized by the node adapter type (vllm vs llamacpp are wired identically via node.client.stub.ForwardPass). Remove run_pipeline_overlap or implement real pipelined overlap; if kept, set batch_size/seq_len consistently with run_pipeline.

---

### F-086 — [Medium] Merkle roots are not comparable across nodes because the digest algorithm depends on optional xxhash

`src/distllm/dist/merkle.py:12` · zone=`dist-exec` · category=`bug`

- **Summary:** The Merkle root and all internal hashes are 16 hex chars when `xxhash` is installed and 64 hex chars when it is not (xxh64 vs SHA-256, and empty-leaf padding of differing length). Since this tree is the cross-node page-table sync primitive (cache_digest.build_merkle_digest/diff_merkle use it to 'diff() between clusters'), two nodes in the same cluster with different dependency sets compute different roots for identical leaves → `diff()` reports every leaf as different → full cache/sync churn, or a node refuses a valid digest. The root format is not pinned per digest version.
- **Evidence (verbatim):**
```
try:\n    import xxhash as _xxhash\n    _USE_XXHASH = True\nexcept ImportError:\n    _USE_XXHASH = False\n_HASH_HEX_LEN = 16 if _USE_XXHASH else 64
```
- **Impact:** Cluster consistency/sync correctness depends on incidental library availability; heterogeneous nodes silently disagree on cache digests, causing redundant transfers or spurious full diffs.
- **Reliability:** Node A runs with xxhash installed (root len 16), node B without (root len 64): same leaves → MerkleTree(leaves).root differ in length → diff() returns all indices.
- **Recommendation:** Pin one algorithm (SHA-256) for the Merkle digest/padding irrespective of whether xxhash is importable, or partition the 'hashed-leaf digest' length from 'content digest' so nodes always agree. Store the chosen algorithm/version in CacheDigestExchange._DIGEST_VERSION and reject mismatches explicitly instead of silently diverging.

---

### F-087 — [Medium] E2E tensor encryption in ForwardPass is dead code — _e2e is always None, tensors transit in plaintext

`src/distllm/dist/node_service.py:199` · zone=`dist-exec` · category=`security`

- **Summary:** NodeServicer wraps most tensor fields in `encrypt_tensor_payload`/`decrypt_tensor_payload`, but `NodeServer.start` (line 448) constructs `NodeServicer(self._worker, cluster_key=...)` with no `e2e_encryption`, so `self._e2e` is always None. Both wrappers return the raw bytes unchanged when e2e is None (e2e.py lines 425-446). There is no code path that ever creates/attaches an E2EEncryption (no key exchange on the worker). The 'encrypt' calls are pure no-ops that give a false sense of encryption while the node channel is protected only by optional TLS + a shared cluster key.
- **Evidence (verbatim):**
```
output_pb.raw_data = encrypt_tensor_payload(output_pb.raw_data, self._e2e)
```
- **Impact:** An operator who reads ForwardPass and believes node-to-node tensors are E2E-encrypted gets plaintext on the wire; any network observer with access to the (unauthenticated-in-the-clear) channel can read hidden states and KV cache.
- **Reliability:** Trace: NodeServer.start → NodeServicer(worker, cluster_key=key) → self._e2e=None → encrypt_tensor_payload(None) returns raw unmodified.
- **Recommendation:** Either remove the dead encrypt/decrypt wrappers (the channel is TLS+cluster-key auth), or actually wire an E2EEncryption into NodeServer.start (init a key-exchange/session at startup) and verify prerequisites at boot. If E2E is a required security control, make NodeServer refuse to start with `_e2e=None` when the config mandates it.

---

### F-088 — [Medium] NodeServer.start silently fails open to plaintext when TLS is requested but cert/key are missing

`src/distllm/dist/node_service.py:466` · zone=`dist-exec` · category=`security`

- **Summary:** `use_tls=True` only produces a secure port if `cert_file` AND `key_file` are also provided; otherwise it silently falls through to `add_insecure_port` on 0.0.0.0. worker.main() defaults `use_tls = not insecure` (TLS requested by default, line 606), so a worker started without cert/key args will bind an unencrypted gRPC port while logging 'TLS enabled'. On top of that, worker.reconnect_to_coordinator hardcodes `use_tls=False` (worker.py line 308), so every coordinator-failover restart downgrades to plaintext regardless of the original TLS config.
- **Evidence (verbatim):**
```
if use_tls and cert_file and key_file:\n    ... self._server.add_secure_port(...)\nelse:\n    self._server.add_insecure_port(f'0.0.0.0:{self._port}')
```
- **Impact:** Node-to-node tensors (hidden states, KV cache, weights) transit over the LAN/Internet in plaintext whenever TLS config is incomplete or on failover, defeating the security claim and exposing model data on the wire.
- **Reliability:** Start worker with --insecure unset and no --tls-cert/--tls-key: use_tls=True but add_insecure_port runs; confirm via `ss -tnlp` plaintext listener (no TLS handshake on connect).
- **Recommendation:** Fail closed: if `use_tls` is true and certs are absent, raise a startup error instead of binding insecure. For failover, preserve the original TLS settings (store `self.use_tls/cert/key` on NodeServer) and reuse them in reconnect rather than hardcoding False.

---

### F-089 — [Medium] Gossip fingerprint is based on an unauthenticated, unused DH exchange

`src/distllm/dist/p2p/gossip.py:812` · zone=`dist-net` · category=`security`

- **Summary:** process_key_exchange accepts any dh_public_key and derives a per-peer key, but verify_message() (lines 300-321) always verifies against self._hmac_key and never reads _peer_hmac_keys — so the DH-derived keys are computed and stored but are dead weight and impart a FALSE sense of peer trust (blue team defends 'authenticated peers' that are actually unverified against any identity). Because the DH exchange is unauthenticated, an attacker can complete a handshake and be recorded as a peer with zero proof of identity; combined with the rekey bug above this is the only 'authentication' mechanism and it is cosmetic.
- **Evidence (verbatim):**
```
shared = pow(int(peer_pub), self._dh_private_key, _DH_PRIME); peer_key = self._derive_hmac_key(shared); self._peer_hmac_keys[peer_id] = peer_key
```
- **Impact:** Misleading trust model: operators believe gossip peers are authenticated by DH when the derived keys are never used; a rogue node is silently admitted as an authenticated peer.
- **Reliability:** Any peer that sends a dh_public_key gets a _peer_hmac_keys entry, yet all message verification goes through _hmac_key only — the per-peer key path is unreachable from verify_message.
- **Recommendation:** Either (a) actually verify peer messages using the per-peer DH key after a real authenticated handshake, or (b) delete the DH machinery and rely solely on the deployment-wide shared key. If kept, bind the exchange to an authenticated challenge (sign the DH pubkey with a key rooted in the shared secret) and enforce CERT-like identity pinning.

---

### F-090 — [Medium] MixedPrecisionAutoTuner yields an all-FP32 (no-op) plan for realistic target quality — never reduces precision

`src/distllm/dist/partition/mixed_precision_tuner.py:184` · zone=`dist-partition` · category=`bug`

- **Summary:** In analyze_sensitivity, best_weight/best_act/best_kv = min(scores, ...) which is always FP32 (score 0.0 baseline). In build_precision_plan, threshold=1-target_quality (0.01 for default 0.99); enumerate(PrecisionMode) starts with FP32 whose score 0.0 <= threshold, so the greedy loop selects FP32 for every layer. Realistic target_quality>=0.9 therefore produces an all-FP32 plan and 'estimated_speedup' of 1.0 — the mixed-precision tuner does nothing. apply() only logs; it never changes runtime kernels.
- **Evidence (verbatim):**
```
best_weight = min(scores, key=lambda m: scores[m])   # always FP32 (score 0.0) ... best = PrecisionMode.FP16 for mode in modes:     if scores.get(mode, 1.0) <= threshold:         best = mode; break   # FP32 passes at default target_quality
```
- **Impact:** The AMP 'tuner' never exploits mixed precision in production settings, so the advertised per-layer precision gains are unrealized and the feature is effectively decorative.
- **Effort:** 4-6 hours
- **Reliability:** Trace: profile_layer scores FP32=0.0 (line 106-107); target_quality=0.99 default -> threshold=0.01; FP32 score 0.0<=0.01 selected first. Line 290 apply() only logs.
- **Recommendation:** Treat FP16 (not FP32) as the baseline and exclude FP32 from the selection loop; iterate modes by descending aggressiveness (INT4->INT8->FP8->FP16) and pick the most aggressive within threshold; wire apply() to actually set runtime dtypes. Add tests asserting INT8/FP8 appear for target_quality<=0.95.

---

### F-091 — [Medium] NetworkCostModel._measure_latency Windows detection is inverted, silently degrading probing to TCP fallback on POSIX

`src/distllm/dist/partition/network_cost_model.py:372` · zone=`dist-partition` · category=`bug`

- **Summary:** _measure_latency sets is_win = host_a.startswith('\\\\') or ':' not in host_a. For any plain hostname (e.g. 'node-a', no colon) is_win is True, so on Linux/macOS it invokes Windows ping flags ['-n','-w'] which fail, and control falls through to a raw TCP connect to hard-coded port 50050. Even when ping exists, real ICMP latency is never used and probing relies on a port that is typically not listening.
- **Evidence (verbatim):**
```
is_win = host_a.startswith("\\\\") or ":" not in host_a                 if is_win:                     cmd = ["ping", "-n", "1", "-w", ...]  # Windows flags on POSIX hostnames
```
- **Impact:** Inter-node latency measurements in the region-aware partitioner are garbage fallback values on POSIX, biasing same-region/affinity decisions.
- **Effort:** 2-4 hours
- **Reliability:** Trigger: probe_once with default hostnames node-0..node-N on Linux; ':' not in 'node-0' => is_win True => ping fails => falls to socket.create_connection((host,50050)) at line 396, which times out -> returns 10.0ms.
- **Recommendation:** Detect OS via sys.platform.startswith('win') (and per-host IPv6 via ':' in host_b) instead of the hostname heuristic; fall back to ICMP/tcp as intended; make the fallback port configurable.

---

### F-092 — [Medium] PartitionOptimizer._finalize reports unquantized latency/throughput/OOM even when a quantization plan is attached

`src/distllm/dist/partition/optimizer.py:226` · zone=`dist-partition` · category=`bug`

- **Summary:** With _enable_quant_tuning, _evaluate_node_with_quant picks the best quant and records _quant_choices, but _backtrack (re-evaluates each point with plain self._cost_model.evaluate) and _finalize (recomputes max_time, throughput, num_oom_nodes from the plain model) ignore the quantized costs. So the returned max_node_time_ms/estimated_throughput/num_oom_nodes are the FP16 numbers even though a quant plan is attached — a node that only fits via quantization is reported as OOM, and the quant benefit is invisible in the headline metrics.
- **Evidence (verbatim):**
```
for pt in points:     cost = self._cost_model.evaluate(pt.node_id, ...)  # plain, no quant     max_time = max(max_time, cost.total_time_ms)     if not cost.fits_in_memory: oom_count += 1
```
- **Impact:** Users comparing DP-with-quant see no latency or memory improvement and may see spurious OOM, defeating the quant-aware DP feature and misinforming cluster sizing.
- **Effort:** 4-6 hours
- **Reliability:** Trace: quantized costs live in _dp via _evaluate_node_with_quant (lines 327-397) and _quant_choices; but _backtrack (198-208) and _finalize (226-253) call self._cost_model.evaluate (analytical) on the same points. quant_partition.py's QuantAwareSolution correctly threads QuantizedNodeCost through finalize — this solver does not.
- **Recommendation:** In _finalize/_backtrack, when _quant_choices has an entry for (node_idx,start,end), use QuantizationAwareCostModel.evaluate_with_quant with the chosen recommendation to compute the reported time, throughput, and fits_in_memory.

---

### F-093 — [Medium] QualityCalibrator._calibrate_method never quantizes — every method scores identical (delta 0), calibration is meaningless

`src/distllm/dist/partition/quant_calibrate.py:172` · zone=`dist-partition` · category=`bug`

- **Summary:** _calibrate_method calls self._compute_perplexity(model, inputs) on the same unquantized model regardless of method (the comment admits 'real impl would use the actual quantization pipeline'), so quant_ppl == baseline for every method, perplexity_delta is always 0, and to_quality_loss_dict yields all-zero losses. The APO can thus treat 4-bit as lossless.
- **Evidence (verbatim):**
```
# Apply quantization to model (simplified -- real impl would use         # the actual quantization pipeline)         quant_ppl = self._compute_perplexity(model, inputs)         delta = quant_ppl - baseline_ppl
```
- **Impact:** Online calibration cannot distinguish quantizations; if the coordinator ever trusts calibration estimates, it will over-quantize and silently degrade output quality.
- **Effort:** 4-8 hours
- **Reliability:** Trace: calibrate() -> _calibrate_method rounds with the identical model, so delta always 0.0; report.recommended_quality_losses all 0 => tuner believes method quality_loss is free.
- **Recommendation:** Apply the actual quantized variant before measuring (route through the backend's quantized forward), or return error for methods without a real quantized path instead of a fake delta of 0; add a test that injects a model with known quality degradation and asserts delta>0.

---

### F-094 — [Medium] profile_layer_precision casts weights to int8/fp8 then runs the fp16 module, causing dtype mismatch / invalid measurements

`src/distllm/dist/partition/quantization_metrics.py:173` · zone=`dist-partition` · category=`bug`

- **Summary:** profile_layer_precision casts each layer's parameters to torch.int8 (or torch.float8_e4m3fn) and then executes module(cast_input). A torch Linear with int8 weights times a half input raises a runtime dtype mismatch (Half vs Char) on most backends, so 'int8' profiling either crashes or is measured after a broken cast. Also torch.float8_e4m3fn is accessed unguarded (line 173/599), raising AttributeError on torch builds without FP8.
- **Evidence (verbatim):**
```
elif prec == "fp8":     target_dtype = torch.float8_e4m3fn  # FP8 E4M3  (unguarded attr) ... param.data = param.data.to(target_dtype)  # int8 weights then run module
```
- **Impact:** The per-layer mixed-precision profiler cannot reliably measure int8/fp8 layers, undermining the 'profile_all' selection path and any quality-capacity decisions built on it.
- **Effort:** 4-8 hours
- **Reliability:** Trigger: profile_all=True with precisions incl. 'int8' on a CUDA model -> module(cast_input) with int8 weights fails/fp16 math wrong. Same code duplicated at quantization_tuner.py:599/608.
- **Recommendation:** Guard float8 with hasattr(torch,'float8_e4m3fn'); for numeric correctness, profile via real quantized kernels/backends or skip int8 measurement with a logged warning instead of casting weights and running fp16 ops; wrap the forward in try/except and record error=... into the profile.

---

### F-095 — [Medium] AutoMixedPrecisionPipeline NF4 maps to float16 with no packing — recommended 4x weight savings are illusory

`src/distllm/dist/partition/quantization_tuner.py:838` · zone=`dist-partition` · category=`bug`

- **Summary:** AutoMixedPrecisionPipeline._parse_dtype('nf4') returns torch.float16 and apply_to_model_weights merely casts params to fp16 (no scale factors stored). Yet the tuner advertises NF4 with QUANT_PROFILES memory_reduction=0.25 (4x) and build_mixed_precision_plan assigns weight_dtype='nf4' for MLP (compression_ratio 4.0). The result claims 4x compression while weights remain full-size fp16 — zero bytes saved, and subsequent fp16 matmuls lose nothing but the memory accounting lies.
- **Evidence (verbatim):**
```
"nf4": torch.float16,  # NF4 weights stored as fp16 with scale factors ... for param in module.parameters(recurse=True):     if param.dtype != target_dtype: param.data = param.data.to(target_dtype)
```
- **Impact:** Memory-headroom decisions based on nf4 plans are wrong; the coordinator may approve partitions or KV tiers that will OOM at runtime.
- **Effort:** 1-2 days
- **Reliability:** Trace: _parse_dtype line 838 -> torch.float16; apply_to_model_weights casts to fp16 (lines 1016-1018); no 4-bit packing or scale tensors are produced. (duplicate copy at quantization_search.py:83).
- **Recommendation:** Either implement real 4-bit packing with scale factors in apply_to_model_weights (producing a torch-format 4-bit representation), or refuse to recommend NF4 until the apply path can honor it, and set compression_ratio consistently with what is actually stored.

---

### F-096 — [Medium] to_proto_tensor reads CUDA memory before the async copy stream is synchronized

`src/distllm/dist/pipeline/serialization.py:59` · zone=`dist-net` · category=`bug`

- **Summary:** For CUDA tensors the device->host copy is issued on a dedicated copy stream with non_blocking=True, then immediately read via numpy(force=True). numpy synchronizes the tensor's own stream, NOT the copy stream (GPUDirectSerializer handles this by calling copy_stream.synchronize(), but to_proto_tensor never does and forward_request serializes hidden_states immediately after). On busy GPUs this race can read stale/partial CPU memory into the TensorProto raw_data. The comment defers sync "to the caller" but no caller synchronizes the copy stream.
- **Evidence (verbatim):**
```
with torch.cuda.stream(copy_stream): t = t.to("cpu", non_blocking=True) # Sync is deferred ...; raw = bytes(memoryview(t.contiguous().view(torch.uint8).numpy(force=True)))
```
- **Impact:** Latent, nondeterministic corruption of GPU-resident hidden states / KV caches during cross-node transfer on CUDA systems, producing wrong inference.
- **Reliability:** Run serialization while another kernel hammers the same device; occasionally the produced raw_data does not match the tensor contents.
- **Recommendation:** Call copy_stream.synchronize() before numpy(force=True) (mirror GPUDirectSerializer.serialize_gpu at compression_negotiation.py:408-409). Add a stress test that round-trips CUDA tensors under concurrent kernels.

---

### F-097 — [Medium] PrefixCache.store() overwrite path never updates _total_memory_bytes, so memory budgeting drifts

`src/distllm/dist/prefix_cache.py:230` · zone=`core-cache` · category=`bug`

- **Summary:** In the production dist/prefix_cache.py PrefixCache, re-storing an existing key replaces kv_data but does not subtract the old entry's bytes or add the new entry's bytes from/to _total_memory_bytes (it only calls move_to_end and bumps access_count). Repeatedly re-hashing a hot prefix with a different-size KV blob causes the memory counter to permanently diverge: under-counting risks the cache exceeding its budget (OOM), over-counting triggers spurious evictions of otherwise-valuable entries.
- **Evidence (verbatim):**
```
if key in self._cache:     self._cache.move_to_end(key)     self._cache[key]["kv_data"] = kv_data     self._cache[key]["access_count"] = ... + 1     ...     return  # _total_memory_bytes not adjusted for replaced blob
```
- **Impact:** Bound/quality drift of the in-memory prefix cache under sustained hot-prefix reuse; can lead to either cache exceeding its 512MB default budget or premature eviction.
- **Effort:** 1-2 hours
- **Reliability:** store('a'*16, big) then store('a'*16, small): _total_memory_bytes keeps big size. store then store larger keeps small size. evict_until_fit only called at the end of the fresh-insert path, so the stale counter governs (over/under) future eviction decisions.
- **Recommendation:** On the overwrite branch subtract entry_bytes(old=self._estimate_entry_memory(self._cache[key]['kv_data'])) and add entry_bytes(new) so _total_memory_bytes stays exact, then call _evict_until_fit(0). Add a test that stores the same prefix twice with different tensor sizes and asserts memory_util matches.

---

### F-098 — [Medium] DistributedPrefixCache.find_best_node matches on full-sequence hash only — true cross-node prefix sharing never happens

`src/distllm/dist/prefix_cache.py:419` · zone=`core-cache` · category=`bug`

- **Summary:** DistributedPrefixCache advertises 30-50% TTFT reduction for shared prefixes, but update_local_prefix() stores ONLY the full-token-sequence hash into _prefix_hashes[hash]->length, and find_best_node() hashes the ENTIRE query and does an exact-hash lookup against -prefix_hashes and _remote_prefixes. Any query longer than the cached sequence (the normal case: cached system prompt + longer user tail) produces a different hash and misses. The sibling CacheIndex.rolling_prefix_hash (dist/cache.py) segmented-hash approach exists and is correct, but DistributedPrefixCache does not use it, so partial/suffix prefix sharing across nodes is effectively non-functional.
- **Evidence (verbatim):**
```
h = 0; for tok in token_ids: h = ((h*31337)+tok) & ((1<<61)-1) local_len = self._prefix_hashes.get(h, 0); ... for node_id,prefixes in self._remote_prefixes.items(): remote_len = prefixes.get(h,0)
```
- **Impact:** The headline cross-node prefix-sharing feature silently degrades to exact-prompt matching; shared system/few-shot prefixes provide no cross-node TTFT benefit, and peers are notified of prefixes that can never be matched.
- **Effort:** 1-2 days
- **Reliability:** update_local_prefix stores _prefix_hashes[hash(full_seq)] = len(full_seq). find_best_node computes hash(full_query) and looks it up exactly; partial prefixes produce distinct hashes => miss.
- **Recommendation:** Adopt segment/rolling hashes like CacheIndex.rolling_prefix_hash(window_size): store {segment_index -> list of (hash,node,prefix_len)} and find_best_node by comparing the longest run of matching leading segment hashes, mirroring CacheIndex.longest_prefix_match. Add a unit test with two prompts sharing a 32-token prefix and assert best_node matches at 32.

---

### F-099 — [Medium] Unknown tenant_ids bypass quota entirely, and served queued requests never consume tokens (multi-tenant quota evasion)

`src/distllm/dist/quota_enforcer.py:160` · zone=`dist-exec` · category=`security`

- **Summary:** Both quota layers fail open for unregistered tenant_ids: `QuotaEnforcer.try_consume` (line 160) and `MultiTenantSLOEnforcer.should_admit` (multi_tenant.py line 183) return True for any tenant_id not in the table. If tenant_id is derived from a client-supplied string/header (as the docstrings imply: 'tenant_id from API key / JWT'), a caller can register under a novel id or a caller without an SLO row gets unlimited throughput. Additionally `QuotaEnforcer.select_next` dequeues a request (line 251) but never calls `q.consume(...)`, so burst credits are not decremented for served queued requests — an accounting leak that also feeds the same-tenant repeat.
- **Evidence (verbatim):**
```
q = self._quotas.get(tenant_id)\nif q is None:\n    return True  # unknown tenants are not rate-limited
```
- **Impact:** Tenant isolation is mostly advisory: any client can dodge throttling by not being registered or by rotating tenant_id, and queued traffic is billed/rate-limited incorrectly.
- **Reliability:** Set quotas for 'acme' only; call try_consume('unregistered-tenant', 10000) → returns True; call enqueue+select_next and observe total_consumed unchanged.
- **Recommendation:** Fail closed for unknown tenants (reject or default to a shared baseline quota) and enforce at the API boundary that tenant_id maps to an authenticated identity, not a client-declared string. In select_next, call `q.consume(min_tokens)` when serving a queued request. Reconcile the two duplicate quota implementations (QuotaEnforcer and MultiTenantSLOEnforcer) into one enforced path.

---

### F-100 — [Medium] Coordinator failover re-registers the worker over plaintext and downgrades the gRPC server to use_tls=False

`src/distllm/dist/worker.py:308` · zone=`dist-exec` · category=`security`

- **Summary:** WorkerNode.reconnect_to_coordinator tears down the existing server and restarts it with a hardcoded `use_tls=False`, discarding whatever the original TLS state was. During failover the node-to-node channel is re-established unencrypted and the subsequent HTTP registration (`_register_with_coordinator`) uses scheme http unless `self.use_tls` was set before the restart — but the gRPC server itself is always insecure after reconnect. This silently undoes TLS/trust on exactly the moment (coordinator loss) when the cluster is most vulnerable.
- **Evidence (verbatim):**
```
self._server = NodeServer(\n    self, port=self.port, cluster_key=cluster_key,\n)\nself._server.start(use_tls=False,  # Will be configured by caller if needed)
```
- **Impact:** After any coordinator failover there is a period where all worker tensors flow in plaintext regardless of baseline TLS policy.
- **Reliability:** Start worker with TLS, call reconnect_to_coordinator(new_host, port) → NodeServer.start(use_tls=False) binds add_insecure_port.
- **Recommendation:** Persist TLS parameters (use_tls, cert_file, key_file) on the WorkerNode and pass them to the restarted NodeServer in reconnect_to_coordinator instead of hardcoding use_tls=False. Set self.use_tls before restart so registration keeps using https.

---

### F-101 — [Medium] BaseToolProvider and model_router API calls never send the api_key, breaking tool discovery/calls on auth-required clusters

`src/distllm/integrations/_common/base_tool_provider.py:77` · zone=`integrations` · category=`bug`

- **Summary:** BaseToolProvider stores `self._client` with an api_key but `discover_tools`, `discover_tools_from_openapi`, and `call_tool` bypass it and use bare `httpx.get/post` with only a Content-Type header. DistLLMModelRouter.discover_models and auto_route similarly send no Authorization. On any cluster where /v1/tools or /v1/models requires a token, discovery silently falls back to default_tools() and call_tool returns an HTTP 4xx error string.
- **Evidence (verbatim):**
```
resp = _retry(                 httpx.get,                 f"{self.base_url}/v1/tools",                 headers={"Content-Type": "application/json"},                 timeout=10,             )
```
- **Impact:** All LangChain/CrewAI tool-provider discovery and routing silently degrades to defaults and calls return errors on secured deployments, defeating the api_key passed into the constructors.
- **Effort:** 1-2 hours
- **Reliability:** __init__ builds DistLLMClientSync with api_key (lines 62-66) yet every direct httpx call in this class and model_router.py omits Authorization.
- **Recommendation:** Route tool/model calls through the already-constructed self._client (or pass Authorization: Bearer <api_key> headers) consistently instead of standalone httpx callers.

---

### F-102 — [Medium] GitLab get_results builds malformed artifact URLs when the default project is used

`src/distllm/integrations/ci/gitlab.py:228` · zone=`integrations` · category=`bug`

- **Summary:** In gitlab.py get_results the artifact URL uses the raw `project` parameter: `f"{self._base_url}/{project}/-/jobs/{job_id}/artifacts/download"` instead of the resolved `proj`. When the caller relies on the constructor's default project (the common case), `project` is None and every artifact link becomes `{base}/None/-/jobs/<id>/artifacts/download` — a broken link in MR reports.
- **Evidence (verbatim):**
```
artifact_urls.append(                     f"{self._base_url}/{project}/-/jobs/{job_id}/artifacts/download"  # noqa: E501                 )
```
- **Impact:** Evaluation reports posted to MRs contain dead download links whenever the default project is used (the documented primary usage). Pipeline status/logging elsewhere correctly uses self._default_project.
- **Effort:** 30-60 min
- **Reliability:** proj resolved at line 196 from project or self._default_project; line 229 interpolates the raw `project` (None when default used).
- **Recommendation:** Use `proj` (the resolved percent-encoded project) in the artifact URL string, e.g. f"{self._base_url}/{proj}/-/jobs/{job_id}/artifacts/download".

---

### F-103 — [Medium] adapter quantize_int8 is a cosmetic no-op: re-dequantizes in place, saves zero VRAM

`src/distllm/models/adapter.py:338` · zone=`core-training` · category=`bug`

- **Summary:** quantize_adapter rounds params to int8, stores metadata, then immediately writes back `(quantized.to(float32) * scale)` to param.data, so the live model keeps float tensors and no VRAM is freed — only quantization rounding noise is introduced. load_adapter never triggers quantize anyway.
- **Evidence (verbatim):**
```
quantized = torch.clamp(torch.round(param.data / scale), -127, 127).to(torch.int8) param.data.copy_((quantized.to(param.device).to(torch.float32) * scale).to(param.dtype))
```
- **Impact:** Adapters advertised as int8 consume identical VRAM and lose precision; multi-tenant density/eviction-headroom claims unmet.
- **Effort:** 3-6 hours
- **Reliability:** quantize_adapter('x') on a 2-param float16 adapter leaves both params float16, altered only by rounding.
- **Recommendation:** Actually store/use int8 params (with per-tensor scale) so adapter density improves, or remove quantize_int8; do not rewrite param.data with rounded copies for metadata only.

---

### F-104 — [Medium] AdapterRouter concurrency: self._lock is always None (torch has no 'lock' attr) leaving routing map unsynchronized

`src/distllm/models/adapter_router.py:73` · zone=`ops-utils` · category=`bug`

- **Summary:** AdapterRouter.__init__ sets self._lock = torch.lock if hasattr(torch,'lock') else None. torch has no public 'lock' attribute, so self._lock is always None and is never used; self._request_adapters (mapping request_id->adapter_id) is mutated concurrently by activate_for_request/clear_request/build_adapter_batch from request threads with no synchronization, risking dict races and inconsistent batch grouping.
- **Evidence (verbatim):**
```
self._lock = torch.lock if hasattr(torch, "lock") else None  # noqa  (line 73)
```
- **Impact:** Under concurrent multi-request serving, adapter->request routing can corrupt or drop mappings, delivering wrong-adapter (or base-model) inference for some requests.
- **Effort:** 1-2 hours
- **Reliability:** hasattr(torch,'lock') is False, so the branch is dead and no lock guards the shared dict that request threads mutate.
- **Recommendation:** Replace with self._lock = threading.Lock() and hold it while reading/writing self._request_adapters in activate_for_request, clear_request, and build_adapter_batch.

---

### F-105 — [Medium] _verify_download_integrity computes SHA-256 against nothing — it can never detect corruption

`src/distllm/models/model_hub.py:418` · zone=`core-training` · category=`bug`

- **Summary:** The integrity verifier hashes each .safetensors file but never compares the digest to any expected value, so it never warns or fails on truncation/bit-rot. The docstring claims it 'logs warnings for files that don't match', but there is no expected-hash source, making the whole method a cosmetic no-op that gives a false sense of download verification (directly relevant to the safetensors corruption thread).
- **Evidence (verbatim):**
```
sha256 = hashlib.sha256() with open(safetensor, "rb") as f:     for chunk in iter(lambda: f.read(8192), b""):         sha256.update(chunk) logger.debug(f"Verified {safetensor.name}: SHA-256={sha256.hexdigest()[:16]}...")
```
- **Impact:** Corrupted shards are silently treated as valid; a truncated/bit-flipped model file passes 'verification'.
- **Effort:** 3-5 hours
- **Reliability:** Corrupt any byte of a cached .safetensors; _verify_download_integrity() still logs 'Verified' and returns without exception.
- **Recommendation:** Read the expected sha256 from the HF metadata (models--<org>--<name>/snapshots/<hash>/...metadata.json or index 'metadata') and compare; log ERROR and fail on mismatch. Wire into download() and download_layer_subset().

---

### F-106 — [Medium] find_optimal_partition never uses the profiled throughput to weight nodes

`src/distllm/models/partition_planner.py:374` · zone=`core-training` · category=`security`

- **Summary:** Throughputs are keyed by `i*(total_layers//num_nodes)` but the weighting loop looks up by the running layer cursor `current`, which only coincidentally matches a key; every node beyond the first gets default weight 1.0, so the 'optimal' partition collapses to a near-equal split and ignores profiling.
- **Evidence (verbatim):**
```
throughputs = {i * (total_layers // num_nodes): prof[2] for i, prof in enumerate(profiles[:num_nodes])} ... fraction = throughputs.get(current, 1.0) / total_throughput
```
- **Impact:** The advertised 'optimal' heuristic degrades to equal split; no throughput optimization is applied.
- **Effort:** 1-3 hours
- **Reliability:** Two profiles with very different throughputs -> both nodes still get ~half the layers.
- **Recommendation:** Key throughputs by each profile's actual (start,end) range and look up by the assigned range, or assign layers proportionally to measured throughput directly.
- **Strategic value:** Platform routes untrusted user model IDs by default; hardening this is tabl

---

### F-107 — [Medium] NTK scaled rope_theta never applied: key 'theta' vs 'rope_theta' mismatch

`src/distllm/models/rope_scaling.py:111` · zone=`core-training` · category=`bug`

- **Summary:** build_rope_scaling_config stores the NTK-scaled base as key 'rope_theta', but apply_rope_scaling checks `if "theta" in rope_config`, which is never true, so model.config.rope_theta is never updated and the model keeps the original 10000 base — silently defeating NTK-aware RoPE beyond the original context length.
- **Evidence (verbatim):**
```
if "theta" in rope_config:     config.rope_theta = rope_config["theta"] # config dict key is "rope_theta", never "theta"
```
- **Impact:** Long-context NTK path is silently broken; positions beyond original max degrade/fail.
- **Effort:** 1-2 hours
- **Reliability:** apply_rope_scaling(model, scaling_type='ntk') leaves model.config.rope_theta at 10000.0.
- **Recommendation:** Use `if "rope_theta" in rope_config:` and set from rope_config["rope_theta"]. Add a test asserting apply_rope_scaling('ntk') updates model.config.rope_theta.

---

### F-108 — [Medium] Valid JWT without a role claim is rejected with 403 instead of granted read-only access

`src/distllm/plugins/auth_plugin.py:417` · zone=`ops-utils` · category=`bug`

- **Summary:** In _validate_jwt_from_context, when a valid JWT lacks a role claim and there is no api_key_role, the fallback returns the string 'read' (line 417) whose comment claims it grants 'minimum access (read-only)'. But 'read' is not a key in ROLE_PRIVILEGES (which contains 'read-only'), so _enforce_rbac later hits 'Unknown role: read' and returns 403, denying the request the code intended to allow.
- **Evidence (verbatim):**
```
return "read"  (line 417); ROLE_PRIVILEGES keys: admin,user-admin,model-admin,auditor,inference-only,read-only (lines 61-68)
```
- **Impact:** Any bearer-JWT client whose token has no recognized 'role' claim cannot get read-only access it is entitled to - functional authorization denial.
- **Effort:** 30 min - 1 hour
- **Reliability:** validate a JWT with no role and no API key; on_request sets api_key_role='read'; _enforce_rbac returns 403 'Unknown role: read'.
- **Recommendation:** Return 'read-only' instead of 'read', matching ROLE_PRIVILEGES and the documented intent.

---

### F-109 — [Medium] MetricsPlugin._error_counts is never written - get_error_counts() always returns empty after server errors

`src/distllm/plugins/builtin.py:229` · zone=`ops-utils` · category=`bug`

- **Summary:** MetricsPlugin initializes self._error_counts as a defaultdict but no method ever populates it: on_error calls _incr('on_error'), which only increments self._hook_counts. As a result get_error_counts() always returns {} even when errors occur, so the 'error/failure rates per plugin' feature it advertises is dead.
- **Evidence (verbatim):**
```
self._error_counts: dict[str, int] = defaultdict(int)  (line 229); on_error -> self._incr("on_error") (line 245); get_error_counts returns dict(self._error_counts) (line 263)
```
- **Impact:** Metrics consumers (dashboards, HealthPlugin error-rate source if used) see zero error counts, hiding real failures.
- **Effort:** 1 hour
- **Reliability:** Trigger on_error; inspect get_error_counts() - returns {} because the defaultdict is never written.
- **Recommendation:** Count errors into _error_counts (e.g. _incr also bump the error map under the same lock, or change on_error to increment _error_counts['on_error']), and add a test asserting non-empty counts after an injected error.

---

### F-110 — [Medium] RateLimitPlugin default counter is shared across all tenants - one tenant can DoS other unconfigured tenants

`src/distllm/plugins/builtin.py:93` · zone=`ops-utils` · category=`security`

- **Summary:** RateLimitPlugin builds a single global _default_counter (builtin.py line 93). on_request falls back to that shared counter for any tenant/model without a per-tenant/per-model override. Therefore one noisy tenant (or a single model) exhausting the default cap blocks every other default tenant from the shared quota, a cross-tenant availability gap - the plugin's own docs claim 'per-tenant rate limiting'.
- **Evidence (verbatim):**
```
self._default_counter = _SlidingWindowCounter(default, window)  (line 93); limiter = self._default_counter when no tenant/model override (lines 122-127)
```
- **Impact:** A single tenant's traffic can starve all other default-quota tenants of capacity, violating the intended per-tenant isolation.
- **Effort:** 2-4 hours
- **Reliability:** Tenant A makes 1000 requests (default cap) -> on_request for tenant B with no override hits the same counter and is rejected 429.
- **Recommendation:** Create a per-tenant counter on first sight (keyed by tenant) when no override exists, instead of a shared default instance, and only use a true global cap if explicitly desired - document that choice.

---

### F-111 — [Medium] PluginRegistry.install() pipes arbitrary pip install command; discover() executes plugin_fn()

`src/distllm/plugins/registry.py:76` · zone=`ops-utils` · category=`security`

- **Summary:** registry.install(package_name) runs subprocess.check_call([sys.executable,'-m','pip','install',package_name]) with no allowlist, index pin, hashing, or sandbox. Combined with discover() calling plugin_fn() after ep.load(), any caller able to reach install or discovery triggers arbitrary third-party package download+execution (setup.py RCE) and arbitrary import-time code - an inherent supply-chain/RCE surface. There is no signature/trust verification of entry-point plugins.
- **Evidence (verbatim):**
```
subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])  (lines 76-78); plugin_fn = ep.load(); metadata = plugin_fn() (lines 38-40)
```
- **Impact:** If install/discover are exposed via API, dashboard, or crafted config, an attacker triggers arbitrary code installation and execution on the coordinator host.
- **Effort:** 3-6 hours
- **Reliability:** Call get_registry().install("a-malicious-name") -> pip executes arbitrary install hook; discover() -> ep.load()+call runs plugin module code at import.
- **Recommendation:** Restrict install to an explicit allowlist, require pypi.org with hash/signature checks, run in a sandbox/non-root user, and reject any package name not matching ^[A-Za-z0-9._-]+$. For discover(), validate plugin identity against a signed index before ep.load(); document the trust boundary (plugins run arbitrary code by design).

---

### F-112 — [Medium] E2E tensor transport fails open to plaintext when PyNaCl is absent or a session isn't established

`src/distllm/security/e2e.py:425` · zone=`core-priv-sec` · category=`security`

- **Summary:** `encrypt_tensor_payload` (module-level wrapper) returns `raw_bytes` UNMODIFIED when `e2e is None` or `not e2e.is_established`, logging one warning. In a federated deployment that requires confidentiality, a node without PyNaCl silently ships raw tensor plaintext, and a peer that expects a session will try to decrypt the plaintext packet and fail. The docstring acknowledges but leaves the downgrade as the default behaviour.
- **Evidence (verbatim):**
```
if e2e is None or not e2e.is_established:     ...logger.warning("E2E encryption not active -- tensor data transmitted in plaintext...")     return raw_bytes
```
- **Impact:** Tensors intended to be confidential can be transmitted unencrypted (or cause a peer decrypt error) without any hard failure.
- **Effort:** 2-4 hours
- **Reliability:** Call encrypt_tensor_payload(raw, e2e=None) -> returns raw plaintext bytes; receive side in a session attempts decrypt of a nonce+ct packet and raises CryptoError.
- **Recommendation:** Add a `require_encryption` flag (already implied by cluster config); when set, raise if the session/PyNaCl is unavailable instead of returning plaintext, and surface it at startup. Keep fail-open only for a documented dev/`allow_plaintext=True` mode.

---

### F-113 — [Medium] validate_http_url validates a resolved IP but returns the original hostname URL (DNS-rebinding TOCTOU if not using safe_urlopen)

`src/distllm/security/utils.py:62` · zone=`core-priv-sec` · category=`security`

- **Summary:** `validate_http_url` resolves the hostname via `socket.getaddrinfo`, validates the resulting IPs against private ranges, then returns the ORIGINAL URL string (hostname-based). A caller that takes `validate_http_url`'s OK and then opens the returned hostname URL (rather than `safe_urlopen`, which pins the validated IP) is exposed to DNS rebinding between validation and connection. `safe_urlopen` is the correct path, but the helper's contract invites misuse.
- **Evidence (verbatim):**
```
if (ip.is_private or ip.is_loopback or ...): raise ValueError(...) return url  # returns hostname URL, not the validated resolved IP
```
- **Impact:** SSRF/DNS-rebinding bypass when a caller follows the validate-then-open pattern instead of safe_urlopen.
- **Effort:** 2-4 hours
- **Reliability:** validate_http_url('https://attacker.example') returns the original URL; a later DNS change re-points attacker.example to 169.254.169.254 before urllib.urlopen connects -> private metadata service reached.
- **Recommendation:** Return or accept the resolved IP (like safe_urlopen) so callers connect to the validated address, or document/reject hostname-based use. Consider returning the resolved-netloc URL and warning callers to use safe_urlopen for actual opening.

---

### F-114 — [Medium] security/watermark.py WeightWatermark stores the whole message in a removable attribute; the fine-tuned weight signal is never used for extraction

`src/distllm/security/watermark.py:232` · zone=`core-priv-sec` · category=`security`

- **Summary:** `WeightWatermark.embed` writes the full payload (`len + message + HMAC tag`) into `module._distllm_watermark` and `extract`/`detect` recover the message purely from that attribute. The 'fine-tune 0.01% of weights toward a target signal' is never read during extraction. Stripping `_distllm_watermark` (or re-saving via torch.save with weights_only/trace) destroys the watermark, so ownership is evidenced only by a plain attribute, not the actual weights.
- **Evidence (verbatim):**
```
object.__setattr__(module, "_distllm_watermark", payload) ...in extract: raw = getattr(module, "_distllm_watermark", None) if raw is None: raise WatermarkError("No watermark found in module")
```
- **Impact:** Ownership verification fails if the attribute is stripped; misleading 'weight watermark' capability.
- **Effort:** 1-2 days
- **Reliability:** Clone a watermarked model, `del model._distllm_watermark`, save; extract() raises 'No watermark found' even though weights still contain the signal.
- **Recommendation:** Implement weight-verifiable watermarking: on extract, recompute the deterministic parameter subset from the secret key and verify the fine-tuned signal (correlation/bit pattern) in the actual weight values, with the attribute only as a hint, not the source of truth. Document that current weight-level 'embedding' is attribute-carried and offers no weight-forensic evidence.

---

### F-115 — [Medium] GumbelWatermark.detect_watermark returns the inverted p-value (lower-tail CDF), misreporting significance

`src/distllm/security/watermark.py:591` · zone=`core-priv-sec` · category=`bug`

- **Summary:** `detect_watermark` computes `p_value = 0.5 * (1.0 + erf(z / sqrt(2)))` — the lower-tail CDF P(Z <= z). For a watermark you want the upper-tail P(Z >= z). For z=4 (strong signal) this returns ~0.999999 instead of ~1e-5, so the reported p-value is the complement of the correct one and misleads users about statistical significance. The `watermark_detected = z_score > 4.0` decision remains correct, so only the reported p_value is wrong.
- **Evidence (verbatim):**
```
z_score = (green_count - expected_count) / std p_value = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))) ...return {"p_value": round(p_value, 6), "watermark_detected": z_score > 4.0}
```
- **Impact:** Misleading detection output; a string with a clear watermark is reported with p ~ 1.0 (not significant) instead of ~0.
- **Effort:** Under 1 hour
- **Reliability:** Injected watermark yields z≈4; current formula -> p≈0.99999 while correct upper-tail p≈1e-5.
- **Recommendation:** Return the one-sided upper-tail p-value: `p_value = 0.5 * (1.0 - math.erf(abs(z_score) / math.sqrt(2.0)))` (or `scipy.stats.norm.sf(z_score)`). Add a test asserting p is near 0 for a heavily watermarked sequence and near 1 for unwatermarked random text.

---

### F-116 — [Medium] gbnf_grammar emits invalid 'value' rule for $ref/$defs and treats nested objects as null

`src/distllm/utils/gbnf_grammar.py:31` · zone=`ops-utils` · category=`bug`

- **Summary:** json_schema_to_gbnf/generate_gbnf_for_json_schema: any property with $ref returns '{name} ::= value', and every $defs entry emits '{name} ::= value' (_extra_rules), but 'value' is not a defined nonterminal in llama.cpp GBNF, so any schema using $ref/$defs (the standard mechanism for recursive/nested/combined schemas) yields a grammar that fails to compile. Arrays and nested objects fall through to root ::= "null". The 'strict' param of generate_gbnf_for_json_schema is ignored.
- **Evidence (verbatim):**
```
def _generate_rule_for_ref(ref_path): return f"{ref_name} ::= value"  (lines 29-31); rules.append(f"{name} ::= value") for $defs (line 98)
```
- **Impact:** Constrained JSON generation from any non-trivial (recursive/ref'd) schema fails; users building structured-output schemas rely on this utility and hit compile errors or get literal 'null'.
- **Effort:** 4-8 hours
- **Reliability:** Feed a schema with '$defs':{'Thing':{...}} and a property '$ref': '#/$defs/Thing'; generated grammar contains an undefined 'value' nonterminal.
- **Recommendation:** Expand referenced defs into real grammar rules (recursively convert each $defs schema and replace $ref with the resolved nonterminal), add array/nested-object support, and honor the 'strict' arg. Add unit tests for a $ref/$defs schema.

---

### F-117 — [Medium] Verification hash-registry compares raw float bytes of logits - guaranteed to mismatch on distributed runs, producing misleading CI signals

`src/distllm/verification/hash_registry.py:64` · zone=`ops-utils` · category=`bug`

- **Summary:** compute_output_hash hashes tensor.float().numpy().tobytes(). The runner stores per-step logits/hidden states in the registry and the CLI prints a 'Hash registry' pass_rate. Because the distributed path applies INT8 quantize/dequant between pipeline stages (runner._run_distributed) and gRPC serialization, raw-byte SHA-256 of logits essentially never matches the reference, so the hash pass_rate reports near-total failure even when the tolerance-based comparator passes. Only token_ids/text hashes are robust.
- **Evidence (verbatim):**
```
arr = tensor.detach().cpu().contiguous(); h.update(arr.numpy(force=True).tobytes())  (lines 63-64)
```
- **Impact:** CI/dashboards relying on the hash pass_rate see false negatives for the exact drift the harness is meant to detect, eroding trust in verification.
- **Effort:** 3-5 hours
- **Reliability:** Reference vs INT8-distributed logits differ in low-order float bits every step; raw-byte digest never equals.
- **Recommendation:** Either hash only token_ids/text (deterministic signals) in the registry, or quantize logits to a coarse bucket (e.g. round to N decimals) before hashing and document tolerance; keep raw-byte hashing only for same-hardware exact-repro. Update runner/CLI to distinguish 'byte-exact' vs 'approximate' comparisons.

---

### F-118 — [Medium] Cluster-key secret file (`~/.distllm/cluster_key`) has no permission hardening and is never created by the CLI

`src\distllm\cli\main.py:572` · zone=`cli` · category=`security`

- **Summary:** The CLI reads the shared cluster auth key from `~/.distllm/cluster_key` (main.py:572-575) but never creates that file or restricts its permissions; a grep across src/distllm/cli finds zero chmod/umask/0o600 usage. The sibling cert path does harden private keys (core/certificate_manager.py), so the CLI's cluster-key handling is an inconsistency that can leak the shared credential on multi-user hosts.
- **Evidence (verbatim):**
```
resolved_key = cluster_key or os.environ.get("DISTLLM_CLUSTER_KEY", "")     if not resolved_key:         key_path = os.path.expanduser("~/.distllm/cluster_key")         if os.path.isfile(key_path):             with open(key_path) as kf:                 resolved_key = kf.read().strip()
```
- **Impact:** The cluster authentication key file is read with no mode check and the CLI never creates it under restricted permissions (unlike cert private keys which are chmod 0o600'd in core/certificate_manager.py). On multi-user systems an overly-permissive `~/.distllm/cluster_key` exposes the shared cluster credential; on Windows there is no permission enforcement at all.
- **Reliability:** Evidence: no chmod/umask/0o600 appears anywhere in src/distllm/cli; cluster_key is only ever read (main.py:570-575 and core/coordinator_cli.py:33) and never written or permission-hardened by the CLI; cert private keys are hardened in core/certificate_manager.py:363-368 (os.chmod 0o600).
- **Recommendation:** When the CLI creates/reads `~/.distllm/cluster_key`, set os.chmod(key_path, 0o600) after write (matching certificate_manager.py line 365) and warn if an existing key has group/world-readable bits (stat.S_IRGRP|stat.S_IROTH). Provide a `distllm security key init` path that generates the key with 0o600 so the documented file is actually created instead of only being read. Audit onboard.py:214 (open(path,'w') for ~/.distllm/config.yaml) the same way since that config can embed API keys.

---

### F-119 — [Medium] federate train reports success but submits a nonexistent adapter file and swallows merge errors

`src\distllm\cli\main.py:767` · zone=`cli` · category=`bug`

- **Summary:** federate train submits a hardcoded, never-created adapter path (`/tmp/distllm-federated/{adapter}.pt`) and swallows coordinator/merge failures with bare `except Exception: pass`, so the command can print 'Federated training complete' without having produced a usable federation artifact. The hardcoded absolute Unix path is also non-portable to Windows.
- **Evidence (verbatim):**
```
"adapter_path": f"/tmp/distllm-federated/{adapter}.pt", ...         except Exception:             pass
```
- **Impact:** The coordinator receives a submit payload pointing at a file that is never written; the federation round is recorded against a non-existent artifact, and the subsequent merge is silently skipped via except-pass, so 'federated training' reports success while performing nothing. On Windows (this repo's platform) the hardcoded /tmp path is invalid.
- **Reliability:** Repro: run `distllm cluster federate train --model X --adapter A --data d.jsonl` — after 'Training locally...' completes, the code posts an adapter_path of `/tmp/distllm-federated/A.pt` (adapter bytes are never written there), and if the coordinator is up the merge POST runs inside `except Exception: pass`. No code in src/distllm writes `/tmp/distllm-federated/*.pt`.
- **Recommendation:** Write the trained adapter to a real path via tempfile (e.g. tempfile.gettempdir()/adapter.pt) or actually persist mgr output, then submit that path; only POST /rounds/submit after the file exists. Replace the bare `except Exception: pass` (lines 786-787) on the merge trigger with a log + non-zero exit on expected coordinator reachability, consistent with the exit-code policy above.

---

### F-120 — [Medium] In _promote_pending the prefill-budget rejection is dead code: a candidate whose chunk exceeds remain_p is accepted

`src\distllm\core\batch_scheduler.py:806` · zone=`core-router-sched` · category=`bug`

- **Summary:** The reject clause `if chunk > remain_p and remain_t - chunk < 0:` (batch_scheduler._promote_pending) can never be true: every accepted candidate already satisfies c_tokens <= remain_t (checked earlier), and chunk <= c_tokens, and when the slack branch runs chunk is clamped to remain_t*(1-slack). So remain_t - chunk >= 0 always, and candidates with chunk > remain_p are promoted anyway, overrunning the per-iteration prefill token budget.
- **Evidence (verbatim):**
```
Earlier: `if remain_slots <= 0 or (remain_p <= 0 and remain_t <= 0) or c_tokens > remain_t: rejected...continue`; then `chunk = c_tokens` (or max_prefill_tokens); then `chunk = min(chunk, int(remain_t*(1-prefill_slack_ratio)))`. So chunk<=remain_t always, making `remain_t - chunk < 0` false and the branch dead.
```
- **Impact:** The prefill-token budget (an anti-decode-starvation control) is silently ignored for the common case, letting prefill eat into decode slots under load.
- **Effort:** 0.5 hours
- **Reliability:** Under a tight prefill budget with a large new prefill request that fits in max_total_tokens, the branch never rejects, and remain_p goes negative while the sequence is promoted, allowing an iteration to exceed its configured prefill-token ceiling.
- **Recommendation:** Replace the condition with the prefill-budget check actually intended, e.g. reject when `chunk > remain_p` (and there is no decode-reserved slack covering it), or clamp chunk = min(chunk, remain_p) instead of rejecting. Add a unit test where remain_p < c_tokens to assert the candidate is not over-billed against the prefill budget.

---

### F-121 — [Medium] SchemaConstrainedDecoder.json_schema(schema)/pydantic()/from_response_format('json_schema') IGNORE the JSON schema entirely — output is constrained to 'syntactic JSON' not the schema

`src\distllm\core\constrained_decoder.py:584` · zone=`core-decoding` · category=`bug`

- **Summary:** constrained_decoder.py's `json_schema()` builds a bare `JSONSchemaFSM()` and never inspects the `schema` argument (only valid-JSON-syntax bytes are allowed); `pydantic(model)` computes `model.model_json_schema()` and then discards it; `from_response_format` for 'json_schema' also returns a generic JSONSchemaFSM. So `response_format={type:'json_schema', schema:{...required fields, enums, types...}}` produces valid-but-arbitrary JSON, not schema-conformant output. The API surface (api/streaming.py, grammar_constrained.py, request_pipeline.py use SchemaConstrainedDecoder) implies schema enforcement that the token-level path never performs.
- **Evidence (verbatim):**
```
def json_schema(self, schema=None):  fsm = JSONSchemaFSM()  return ConstrainedConstraint(fsm, self._token_index)
```
- **Impact:** Users relying on json_schema response_format for field/enum guarantees get arbitrary JSON wrapped in valid syntax; schema validation then either passes trivially (if shallow) or fails at post-validation, wasting the response.
- **Reliability:** Code trace: json_schema()/pydantic() (lines 584-619) create JSONSchemaFSM() with no schema arg. from_response_format('json_schema') (lines 639-644) ignores response_format['schema']. JSONSchemaFSM.get_allowed_bytes only encodes generic JSON grammar.
- **Recommendation:** Compile the schema into the FSM: derive required keys/enum values/type-based byte classes from the schema before building the constraint (mirror structured_output.validator behavior but at token level). Alternatively, explicitly document and reject: if a non-empty schema is passed, raise NotImplementedError like the grammar branch does, so callers do not assume conformance that is not guaranteed. Add a test asserting that a {required:['k']} schema cannot generate an object missing 'k'.

---

### F-122 — [Medium] GBNFFSM/GBNFParser in grammar_decoder.py is not a real grammar FSA: character classes make the mask a silent no-op (all tokens allowed)

`src\distllm\core\grammar_decoder.py:226` · zone=`core-decoding` · category=`security`

- **Summary:** GBNFFSM compiles a grammar into a single literal `_target` string via `_extract_target`, which only takes the FIRST alternative and only literal tokens; `_resolve_rule` drops character-class and group tokens (returns '') . For the ubiquitous `root ::= number` / `digit ::= [0-9]` grammar the resolved target is empty, so `get_allowed_bytes()` returns an empty set and `get_logits_mask` returns `torch.ones(...)`: ALL tokens allowed — the 'constrained' decode is completely unconstrained. It also cannot express alternatives (`"hello" | "hallo"` only accepts `hello`) and its token mask requires every byte of a token to equal a single allowed byte, so it can only ever emit single-char tokens. It is currently orphaned (SchemaConstrainedDecoder.from_response_format raises NotImplementedError for grammars; no production importer imports it), so risk is latent, but the module and tests ship as a working decoder.
- **Evidence (verbatim):**
```
if not allowed_bytes: return _torch.ones(vocab_size, dtype=_torch.bool)  ...  if decoded and all(ord(b) in allowed_bytes for b in decoded): mask[token_id] = True
```
- **Impact:** A maintainer wiring grammar_decoder into production today would see 'grammar-constrained' output that is not grammar-constrained at all, silently, for the most common regex-ish GBNF grammars.
- **Reliability:** Repro: GBNFFSM('root ::= "hello" | "hallo"').parse() -> root=[['"hello"'],['"hallo"']]; _extract_target uses root[0] only -> target='hello'; get_allowed_bytes at pos0={ord('h')} so 'hallo' never generateable. GBNFFSM('digit ::= [0-9]\nroot ::= digit+') -> _resolve_rule('digit') yields '' (no quoted literal), target='', get_logits_mask returns all-ones (unconstrained).
- **Recommendation:** Either implement a real GBNF -> NFA/DFA (the repo already has grammar_constrained.gbnf_to_regex for a subset — delegate to it and to outlines when available) or delete grammar_decoder.py and rely on the explicit NotImplementedError so grammars never silently no-op. At minimum, make get_logits_mask fail loudly when the grammar cannot be represented (char classes present) instead of returning an all-ones mask.

---

### F-123 — [Medium] Regex ReDoS via user-supplied workload patterns in _classify_workload/_compute_confidence

`src\distllm\core\model_router.py:753` · zone=`core-router-sched` · category=`security`

- **Summary:** model_router.py guards only its own _match_rule regexes with a coarse length+quantifier heuristic (len>100 and `(.*[+*].*)[+*]`). But _classify_workload and _compute_confidence run every regex in _workload_patterns (extendable by operators via add_workload_patterns or config) with no guard and no timeout. A catastrophically-backtracking pattern like (a+)+S added to the workload patterns DoSes the router thread per request. The existing guard itself is ineffective for patterns under 100 chars.
- **Evidence (verbatim):**
```
_match_rule (line 723): `if len(rule.pattern) > 100 and re.search(r'\(.*[+*].*\)[+*]', rule.pattern): ... return False` then `re.search(rule.pattern, text, re.IGNORECASE)`. _classify_workload (753): `for rx in patterns.get("regex", []): if re.search(rx, text_lower, re.IGNORECASE)` — no guard. _compute_confidence (782-796) runs re.findall with no guard either.
```
- **Impact:** CPU-exhaustion denial-of-service on the routing hot path from a single crafted prompt or operator-supplied pattern; the existing 100-char heuristic does not cover short patterns.
- **Effort:** 1-1.5 hours
- **Reliability:** Operator calls router.add_workload_patterns('x', regex=[r'(a+)+$']); a request with many 'a's then runs re.search(re) in _classify_workload, taking exponential time (classic catastrophic backtracking) and pinning a core per request → facility-wide latency spike / DoS.
- **Recommendation:** Run ALL user/config-supplied regexes on re.compile with a regex timeout or apply the ReDoS guard uniformly in a helper used by _match_rule, _classify_workload, and _compute_confidence; reject/back-off patterns matching `(…+)+`, unterminated alternation, or >some-length nested quantifiers (e.g. ReDoS-salient signatures) before search. Precompile patterns once. Add adversarial regex tests (NFA blowup patterns) asserting bounded latency.

---

### F-124 — [Medium] MultiDraftVerifier.generate, TreeMultiDraftVerifier, and PipelinedSpeculativeDecoder sample a correction token WITHOUT proper distribution-preserving resampling after rejection

`src\distllm\core\multi_draft_verifier.py:178` · zone=`core-decoding` · category=`bug`

- **Summary:** In several verifiers the correction token after the first mismatch is sampled directly from the target logits at the rejection position (`next_logits = target_logits[:, generated.shape[1]-1, :]` then `_sample`), which is the 're-sample from target' approximation rather than the spec-decoding-correct correction `(p - q)/(1 - q)` distribution from Leviathan et al. This biases the emitted sequence away from the true target distribution under rejection sampling — the acceptance is only correct if the correction token is drawn from max(0, p-q)/(1-q). distributed_speculative.py and speculative_decoder.py document the proper min(1, p/q) draw but pair it with a naive target resample, so the serialized distribution is not preserved. The proper guard (q<=0 -> reject) is present, but the correction is not.
- **Evidence (verbatim):**
```
next_token = self._sample(next_logits)  # sampled straight from target logits, no (p-q)/(1-q) correction
```
- **Impact:** Long-run token distribution drifts from the target model; for self-speculative/MTP/tree paths this silently changes model behavior (repetition, grammar) even when acceptance counts look correct.
- **Reliability:** Conceptual: standard spec-decoding correction uses max(0,p-q)/(1-q); code resamples p only (e.g. SelfSpeculativeDecoder._verify_tokens fallback line 468 `if rand >= p: return i` then _sample(next_logits)).
- **Recommendation:** On rejection, sample the correction token from the normalized `max(0, p - q)` distribution (subtract the draft's probability from the target's and renormalize), using the draft_probs/target_probs already computed in the verifier. Add a unit test comparing the empirical next-token marginal under this decoder against the target's marginal to prove distribution preservation.

---

### F-125 — [Medium] MultimodalEngine silently discards image/audio/document tensors — non-text inputs are replaced by placeholder text

`src\distllm\core\multimodal_engine.py:184` · zone=`core-gen-rag` · category=`bug`

- **Summary:** MultimodalEngine.process builds `prompt = _build_multimodal_prompt(text, modality)` which only prepends the strings '[IMAGE]'/'[AUDIO]'/'[DOCUMENT]' to the text and sends that string to the coordinator. The actual image/audio/document_pages tensors are never forwarded, and the encoder nodes set via set_vision_encoder_node/set_audio_encoder_node/set_document_processor_node are never read inside process().
- **Evidence (verbatim):**
```
if modality == ModalityType.IMAGE: return f"[IMAGE] {text}" ... result_text = self._coordinator.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
```
- **Impact:** Callers believe they performed vision/audio/document inference but the model only ever saw a '[IMAGE]' marker; incorrect results with no warning. Tests/core/test_multimodal_engine.py assert on the marker/prompt shape, reinforcing the stub.
- **Reliability:** process(image=img_tensor, text='describe') keeps modality=IMAGE but line 168 builds only the text prompt and never passes img_tensor to the coordinator; _vision_encoder_node is set but never referenced outside the setter. The multimodal branch is a text-echo of a marker string — GIGO.
- **Recommendation:** Either forward the real tensors+embeddings into a multimodal-capable generate path (e.g. route through ModalityEncoder/Voyager), or fail loudly with NotImplementedError when an encoder node/tensor path is requested instead of fabricating textual markers.

---

### F-126 — [Medium] RequestLatencyTracker._completed grows without bound (memory leak)

`src\distllm\core\request_latency.py:71` · zone=`core-router-sched` · category=`bug`

- **Summary:** complete() appends every finished RequestLatencyInfo to self._completed forever. Consumers (get_recent_metrics, get_sla_percentiles) only ever read the last 50-100 entries, so the historical tail is never needed, yet the list accumulates one entry per completed request for the life of the process.
- **Evidence (verbatim):**
```
complete(): `info = self._requests.pop(request_id, None); if info: ... self._completed.append(info)` — no cap. get_sla_percentiles(): `recent = self._completed[-window_size:]`; get_recent_metrics(): `self._completed[-limit:]`. Only a window is read.
```
- **Impact:** Per-request metadata (ids, timestamps, token counts) accumulates indefinitely on long-running servers; stable-state heap grows linearly with total served requests.
- **Effort:** 0.25 hours
- **Reliability:** Register+complete 1M synthetic requests observed memory ids.len grows to 1M entries while only the trailing 100 are used.
- **Recommendation:** Bind the list, e.g. keep only the last N (say 5000) completed entries in complete(): `self._completed.append(info); if len(self._completed) > self._max_completed: self._completed = self._completed[-self._max_completed:]`. Add a test asserting _completed is bounded after many complete() calls.

---

### F-127 — [Medium] resource_manager._tcp_health_check claims a zero-byte send but performs no probe

`src\distllm\core\resource_manager.py:390` · zone=`core-router-sched` · category=`bug`

- **Summary:** The docstring and inline comment promise 'Verify connection is alive with a zero-byte send', but the method only does pool.get() + settimeout() + pool.put() and returns True. A stale socket already in the connection pool is returned 'healthy' with no actual connectivity verification, so health checks can report a dead node as healthy.
- **Evidence (verbatim):**
```
`sock = self._conn_pool.get(host, port); sock.settimeout(timeout); self._conn_pool.put(host, port, sock); return True` with `except (OSError, ConnectionError, socket.timeout): ... return False`. No send/recv of any byte occurs.
```
- **Impact:** Could keep requests routed to a dead node or fail failover, since a stale-but-pooled connection is trusted as a health signal.
- **Effort:** 0.5 hours
- **Reliability:** A pooled TCP socket whose peer has silently closed (no RST observed yet) will be returned by pool.get() and put back; _tcp_health_check returns True though the node is unreachable. This path is the fallback health check in health_check_all/health_check_all_async.
- **Recommendation:** Perform a real check before declaring healthy, e.g. `sock.sendall(b'')` + optional `sock.recv(0)` or a getpeername()/SO_ERROR probe, and only then pool.put(). If ConnectionPool.get() already opens a guaranteed-fresh socket, get the pool behavior to confirm; otherwise add an explicit liveness call. Add a test with a closed/broken socket already pooled and assert healthy=False.

---

### F-128 — [Medium] MultiDraftVerifier consensus rejection-sampling re-derives q from ONLY draft_forwards[0], but the accepted token was chosen by agreement of ALL drafts

`src\distllm\core\speculative_decoder.py:653` · zone=`core-decoding` · category=`bug`

- **Summary:** speculative_decoder.py MultiDraftSpeculativeDecoder._verify_tokens builds the draft by requiring all draft models to predict the identical token (consensus), yet the per-position rejection sample uses `self._draft_forwards[0]`'s probability as q (line 654: `draft_out = self._draft_forwards[0](shared_input)`). The real draft probability of a consensus token is the joint/maximum over the agreeing models; using a single model's q misstates the acceptance ratio min(1,p/q), so the acceptance/rejection draw is biased and the correction token distribution is wrong (fails to reproduce the target distribution exactly). This is the wired multi-draft path (inference_engine imports MultiDraftSpeculativeDecoder), so it is the production multi-draft behavior.
- **Evidence (verbatim):**
```
draft_out = self._draft_forwards[0](shared_input, **kwargs) ... q = draft_probs[0, consensus_tokens[0, i]].item()
```
- **Impact:** Systematically biased rejection sampling for the multi-draft strategy: over-accepts/over-rejects and the produced distribution diverges from the target's, so multi-draft output quality and acceptance stats are unreliable even though the path is wired.
- **Reliability:** Code trace: _generate_consensus_draft returns only token IDs + length (no probabilities); _verify_tokens re-runs draft_forwards[0] to get q for a token that required model[2] etc. to agree.
- **Recommendation:** Track the draft probability at consensus-build time: when all models agree on token t at position i, record q as the max (or product-normalized) probability of t across the agreeing models and pass it into _verify_tokens as draft_logprobs (the class already supports a draft_logprobs arg on the other decoders). This avoids a second draft call AND uses the true consensus probability.

---

### F-129 — [Medium] JSONSchemaConstraint.get_logits_mask (the PRODUCTION structured-output path) masks on FIRST character only, so tokens whose later bytes break JSON structure pass and malformed JSON can be emitted

`src\distllm\core\structured_output\__init__.py:160` · zone=`core-decoding` · category=`security`

- **Summary:** The constraint wired into the API (chat.py, chat_service.py, completion_service.py, token_generator.py sample_batch/apply_constraint, coordinator_request.py, request_pipeline.py) builds a mask by comparing each token's FIRST character ord against the set of valid first chars for the current JSON state (`is_valid = (first_ords.unsqueeze(1) == valid_ords_dev.unsqueeze(0)).any(dim=1)`). It never walks the full token byte sequence through the FSM. Tokens like `}x`, `,hello`, `123abc`, or `"v"junk` start with a legal byte (respectively `}`, `,`, `1`, `"`) and are masked allowed, but their later bytes create invalid JSON that the character-level `_transition` silently accepts (it cannot reject). The docstring even over-claims: 'ensures the output is valid JSON syntax.' Producing invalid JSON for consumers that parse it is a prompt/response integrity issue.
- **Evidence (verbatim):**
```
first_ords = torch.tensor([ord(c) if c else 0 for c in first_chars]...)  is_valid = (first_ords_dev.unsqueeze(1) == valid_ords_dev.unsqueeze(0)).any(dim=1)  mask[:n] = is_valid
```
- **Impact:** Structured responses labeled json_object/json_schema can be returned as invalid JSON, breaking downstream parsers/agents that trust the schema guarantee; inconsistent with the framework integrations which rely on valid JSON.
- **Reliability:** Attack trace: set state to after_value (valid first chars {,}), tokenizer returns token `,hello`. first_ord=ord(',') is in the allowed set -> mask[tid]=True. Sequence sample such that the model picks it; _transition(',') -> after_comma, then 'h','e','l' -> returns same state, but the emitted JSON now has a bare `,hello` where a key was expected, which json.loads rejects.
- **Recommendation:** Validate the WHOLE token before allowing it: for each candidate token id, feed every byte of `tokenizer.decode([tid])` through the FSM transition and require the token to be a valid FSM prefix (and that either it lands in an accepting state or a legal continuation exists). Cache results per (state, token_id). This is the cost of sound constrained decoding; gate the slow full-token walk behind the existing `_mask_cache`. Follow constrained_decoder.py's GBNF get_logits_mask which already does 'all bytes valid' checking (grammar_decoder.py line 237).

---

### F-130 — [Medium] SchemaValidator.validate treats booleans as integers/numbers (isinstance(True,int) is True) and ignores enum/anyOf/const/min/max/format — json_schema validation is unsound

`src\distllm\core\structured_output\validator.py:88` · zone=`core-decoding` · category=`bug`

- **Summary:** structured_output/validator.py `_check_type`: for type 'integer' it uses `expected=(int)`, so `isinstance(True, int)` is True and a JSON `true` scalar passes an 'integer' schema check (Python's bool IS an int subclass); same for 'number'=(int,float). The validator also never checks `enum`, `anyOf`/`oneOf`, `const`, numeric `minimum`/`maximum`/`multipleOf`, `minItems`/`maxItems`, `minLength`, `pattern`, or `additionalProperties:false` — so a schema-requiring `enum:['a','b']` accepts 'x', and `additionalProperties:false` (config `SchemaConfig.allow_additional_properties=False` which is never consulted) accepts unknown keys. validate_structured_output() (structured_output/__init__.py) falls back to ``jsonschema`` when installed, masking the gap only when that dependency is present.
- **Evidence (verbatim):**
```
type_map = {... 'integer': int, 'number': (int, float) ...}  expected = type_map.get(expected_type); if expected is None: return True; return isinstance(data, expected)
```
- **Impact:** Output that violates the schema's enum/bounds/additional-property constraints is marked valid, so invalid structured data reaches consumers and the correctness flag `valid=True` is wrong.
- **Reliability:** Repro: SchemaValidator().validate(json.loads('true'), {'type':'integer'}) -> _check_type(True,'integer') -> isinstance(True,int)=True -> no error -> valid=True. Enum case: validate({'x':'c'}, {'type':'object','properties':{'x':{'enum':['a','b']}}}) returns valid (enum never checked).
- **Recommendation:** Use the jsonschema library when available (as validate_structured_output does) and only run this hand-rolled checker as a fast-path guard. At minimum: special-case bool before int (``if isinstance(data,bool) and expected is int: return False``) and add enum/anyOf/oneOf/const/numeric-bounds/array-length handling, plus honor allow_additional_properties against schema.properties. Add tests that `true` fails integer and that enum-violating output is rejected.

---

### F-131 — [Medium] DisaggregatedRouter load counters are mis-accounted on pool fallback

`src\distllm\core\unified_router.py:388` · zone=`core-router-sched` · category=`bug`

- **Summary:** When a PREFILL request falls back to the DECODE pool (or vice versa), route() adds the load to the fallback_pool load dict, but release() decrements based on the requested phase — so the fallback pool's load is never decremented and the preferred pool's load is falsely decremented. Over time preferred-pool loads go negative/zero while the fallback pool's load grows, skewing least-loaded selection.
- **Evidence (verbatim):**
```
route() fallback: `if fallback_nodes: ... best = min(fallback_nodes, key=lambda n: fallback_load.get(n, 0)); fallback_load[best] = fallback_load.get(best, 0) + 1` but release(): `if phase == RequestPhase.PREFILL: self._prefill_load[node_id] = max(0, ... - 1) else: self._decode_load[node_id] = ... - 1`. The phase/load dict used at allocation and release are decoupled.
```
- **Impact:** Least-loaded routing becomes wrong in disaggregated deployments that allow fallback; nodes in the fallback pool get stuck with inflated load and are skipped.
- **Effort:** 0.5-1 hours
- **Reliability:** With _prefill_nodes=[] and _decode_nodes=['d0']: route(PREFILL) → falls back, d0 decode-load=1. release('d0', PREFILL) decrements _prefill_load (which has no 'd0', stays 0) and leaves _decode_load['d0']=1 permanently, so d0 is never selected again until its pool load hits max.
- **Recommendation:** return (node_id, actual_pool_used) from route() so release() decrements the dict that was actually incremented, OR track load per node in a single dict independent of phase with the pool stored on the node. Add a test: prefill request falls back to decode pool, then release(PREFILL, node) must drop the decode-pool counter.

---

### F-132 — [Medium] Milvus provider uses a process-global fixed connection alias, so two stores interfere and closing one breaks the other

`src\distllm\core\vectorstore\providers\milvus.py:59` · zone=`core-gen-rag` · category=`security`

- **Summary:** _MilvusStore._ensure_collection calls `connections.connect(alias='distllm_default', ...)` every time a collection is created. pymilvus keys connections by alias, so a second _MilvusStore instance (different collection) overwrites the first store's connection under the same alias; close() then `connections.disconnect(alias='distllm_default')` tears down the connection both instances depend on. The `self._connections` list is appended but never used for cleanup, so it is misleading state.
- **Evidence (verbatim):**
```
conn = connections.connect(alias="distllm_default", uri=self._uri, token=self._token) self._connections.append(conn) ... connections.disconnect(alias="distllm_default")
```
- **Impact:** With multiple collections/instances (e.g. per-tenant isolation) clients silently cross-talk or fail after a close; the cleanup path releases a shared resource it does not own.
- **Reliability:** Instantiate _MilvusStore('collA') and _MilvusStore('collB') in one process. collB's _ensure_collection reconnects alias 'distllm_default' targeting its own uri/collection, clobbering collA's connection; collA.close() or any collA use then hits the wrong/closed connection.
- **Recommendation:** Give each _MilvusStore a unique alias (e.g. f'distllm_{id(self)}') or pass aliases explicitly, and disconnect the instance's own alias in close() rather than the shared globals; track handles in self._connections and disconnect from that list.

---

### F-133 — [Medium] Qdrant provider delete() returns the server UpdateResult.status enum, not the number of records deleted (interface contract violation)

`src\distllm\core\vectorstore\providers\qdrant.py:159` · zone=`core-gen-rag` · category=`bug`

- **Summary:** _QdrantStore.delete returns `op_info.status` in both the ids and metadata_filter branches. The base interface documents delete() 'Returns number of records deleted', and callers like RAGPipeline.delete forward this value. Qdrant's UpdateResult.status is a server status enum (e.g. acknowledged/completed), not a count — so RAG deletion reports 0/1 regardless of how many records were removed. Additionally upsert() raises ValueError for namespaces while query()/delete() silently accept and ignore the namespace arg, an inconsistent seam.
- **Evidence (verbatim):**
```
op_info = client.delete(collection_name=self._collection, points_selector=ids) return op_info.status  # type: ignore[return-value]
```
- **Impact:** Deletion bookkeeping in the RAG layer is wrong and inconsistent across providers (pinecone returns count via resp, weaviate returns count, qdrant returns status). A consumer that checks 'was anything deleted' gets a false result.
- **Reliability:** Call store.delete(ids=['a','b','c']) against a collection holding 3 points → qdrant returns UpdateResult.status, which is a status code, so the caller sees 0/1, not 3. The vectorstore provider suite (tests/core/vectorstore/) covers only base/chroma/pgvector/legacy qdrant_store, never this factory-registered provider, so the violation is untested.
- **Recommendation:** Return the actual count for qdrant.delete — query the count or return the number of ids passed (and document the approximation), and add a provider test asserting the return value semantics. Also either honor or reject namespace uniformly in query/delete to match the upsert ValueError.

---

### F-134 — [Medium] Tauri chat and benchmark call the REST API with no Authorization header while the admin layer uses a bearer token

`tauri/src/lib/api.ts:111` · zone=`sdk-arch` · category=`security`

- **Summary:** Tauri's streamChatCompletion and runBenchmark (api.ts) fetch ${baseUrl}/v1/chat/completions directly from the webview with only Content-Type, no auth header. The Rust admin/cluster commands, by contrast, do send state.auth_token as Bearer on /admin/v1/* calls. If /v1/chat/completions enforces the spec-declared BearerAuth, desktop chat and benchmark fail with 401; auth is also never propagated from the token the Rust side manages.
- **Evidence (verbatim):**
```
const response = await fetch(`${baseUrl}/v1/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
```
- **Impact:** Desktop chat/benchmark break (401) when the server enforces auth; inconsistent auth handling between the webview REST path and the Rust command path.
- **Effort:** 2-4 hours
- **Reliability:** api_client.rs sends 'Authorization: Bearer {token}' for admin calls while api.ts chat fetch sends only Content-Type; spec declares 'security: BearerAuth'.
- **Recommendation:** Thread the state.auth_token into streamChatCompletion/runBenchmark headers (or route these calls through Tauri commands like the admin layer) so chat honors the same bearer auth the cluster commands use.
- **Strategic value:** auth_parity

---

### F-135 — [Low] vscode extension makes raw fetches with no API-key support; apps default to a hardcoded 'sk-noauth' bearer credential

`extensions/vscode/src/modelsApi.ts:25` · zone=`sdk-arch` · category=`security`

- **Summary:** The vscode extension's fetchModels (modelsApi.ts), sendToModel, fetchHealth, and fetchMetrics send no Authorization header and expose no apiKey config, so against an auth-enabled server it silently shows empty/errors. Separately, apps/chat and apps/rag default DISTLLM_API_KEY to a hardcoded 'sk-noauth' placeholder that is sent as Bearer.
- **Evidence (verbatim):**
```
const resp = await fetch(url, { signal: AbortSignal.timeout(10_000) });  // no auth; apps/chat/app.py:29 'API_KEY = os.environ.get("DISTLLM_API_KEY", "sk-noauth")'
```
- **Impact:** Against auth-protected servers the extension and apps fail or leak a well-known placeholder bearer value as a de-facto credential.
- **Effort:** 2-3 hours
- **Reliability:** vscode fetches carry only Content-Type; apps/chat/app.py:29 and apps/rag/app.py:33 default API_KEY to 'sk-noauth' and send it as Bearer via openai SDK.
- **Recommendation:** Add an apiKey config to the vscode extension and send 'Authorization: Bearer <key>' on all fetches (keep https-for-non-localhost enforcement). Default apps to empty key (no header) unless set, and warn instead of shipping a known placeholder Bearer credential.
- **Strategic value:** auth_parity

---

### F-136 — [Low] block_eviction_policy: only LRU honors ref_count; LFU/2Q/ARC pick_victim can evict in-use blocks

`src/distllm/core/block_eviction_policy.py:126` · zone=`core-cache` · category=`bug`

- **Summary:** In core/block_eviction_policy.py, LRUPolicy.pick_victim gates victims on 'block_usage[bid].ref_count <= 0' and falls through to the oldest only if none match. LFUPolicy, FIFOPolicy, TwoQPolicy, and ARCPolicy pick_victim iterate tracked/queue/queues and return any block in block_usage, ignoring ref_count — so a block still referenced by an active sequence can be chosen for eviction, corrupting paged KV. Module is currently dead (0 importers), but it is the nominal eviction engine for block pools.
- **Evidence (verbatim):**
```
victim = min((bid for bid in self._counts if bid in block_usage), key=lambda b: self._counts.get(b,0), default=None) # (LRU alone checks block_usage[bid].ref_count <= 0)
```
- **Impact:** Evicting a referenced block yields dangling/incorrect paged KV for live sequences; latent today since the module is not imported.
- **Effort:** 1-2 hours
- **Reliability:** LFU pick_victim selects min-count block among 'bid in block_usage' with no ref_count check; same for 2Q (a1in then am) and ARC (t1 then t2).
- **Recommendation:** Standardize the guard: always require ref_count <= 0 (and skip non-zero refs) in pick_victim across LFU/2Q/ARC/FIFO, mirroring LRUPolicy; add a unit test that a ref_count=1 block is never selected while ref_count=0 peers exist. If the module stays dead, delete instead.

---

### F-137 — [Low] cache_eviction.SemanticGrouping uses process-randomized str hash() for MinHash, so fingerprints are non-deterministic across nodes

`src/distllm/core/cache_eviction.py:121` · zone=`core-cache` · category=`bug`

- **Summary:** core/cache_eviction.py:SemanticGrouping.compute_signature uses hash(f"{token}_{i}"): Python string hashing is salted per-process (PYTHONHASHSEED), so the same token sequence produces a different signature on different nodes/restarts. If distributed warmers ever compared these fingerprints across peers (the stated intent: batch eviction + cache warming grouping), matches would silently fail. The far-superior, deterministic dist/cache.py SemanticGrouping (which uses integer hash(tok) and an LSH index) already exists and is the production copy. Flag: delete the core duplicate and standardize on dist.
- **Evidence (verbatim):**
```
h = hash(f"{token}_{i}") & 0xFFFFFFFF min_hash = min(min_hash, h)
```
- **Impact:** Cache-warming/eviction grouping is non-deterministic across the pool; fingerprint persistence would be corrupted on reload. Latent because module is dead.
- **Effort:** 2-3 hours
- **Reliability:** Two Python processes with different PYTHONHASHSEED compute different hash(f"{token}_{i}") values for identical tokens -> signatures differ -> _similarity degrades to 0 -> groups never match cross-node.
- **Recommendation:** Replace with the deterministic integer signature from dist/cache.py SemanticGrouping (hash(tok) & mask, LSH bands) and drop core/cache_eviction.SemanticGrouping. Never persist or ship fingerprints generated from str hash across processes.

---

### F-138 — [Low] AWS static fallback lists spot price above on-demand (negative discount) — misleads arbitrage/get_cheapest

`src/distllm/core/pricing_providers.py:211` · zone=`core-perf-obs` · category=`bug`

- **Summary:** AWSPricingProvider._fallback() sets p3.2xlarge on_demand=3.06 but spot=3.83, and g5.xlarge on_demand=1.006 but spot=1.41 — spot is MORE expensive than on-demand. spot_discount_pct then computes a negative discount, and PricingManager.get_cheapest(prefer_spot=True) picks bogus pricing. These stale spot values corrupt the arbitrage/cost-comparison features when the live API is unreachable (the common fallback path).
- **Evidence (verbatim):**
```
static = {'p4d.24xlarge': 32.77, 'p3.2xlarge': 3.06, 'g5.xlarge': 1.006, ...} spot_static = {'p4d.24xlarge': 14.40, 'p3.2xlarge': 3.83, 'g5.xlarge': 1.41, ...} # spot_discount_pct -> (1 - 3.83/3.06)*100 = negative
```
- **Impact:** When the AWS price API is unavailable (firewalled/CI), get_cheapest and the dashboard report inflated spot prices and negative discounts, so cost arbitrage and savings estimates are wrong at exactly the times users rely on the fallback.
- **Effort:** 1-2 hours
- **Reliability:** Unit-test fetch via AWSPricingProvider._fallback(): for p3.2xlarge, spot_price(3.83) > on_demand_price(3.06), spot_discount_pct < 0. Repros without network.
- **Recommendation:** Correct the stale static spot values (or clamp spot<=on_demand in the build loop), and add a sanity guard in InstancePricing/spot_discount_pct that returns 0 when spot >= on_demand. Add a unit test asserting no fallback entry has spot_price > on_demand_price.

---

### F-139 — [Low] RadixTreeCache evicts only leaf nodes and double-counts memory on re-store (orphaned alt implementation)

`src/distllm/core/radix_tree_cache.py:96` · zone=`core-cache` · category=`bug`

- **Summary:** The (dead) core/radix_tree_cache: _find_lru_leaf recurses to node with no children and only a LEAF with kv_data is returnable, so a prefix stored at an internal node that also has children is never evictable; evict_lru's while-_count_entries loop can fail to converge if leaves carry no kv_data (the break is inside the kv_data-None branch, but _count_entries() still counts them). Also store() adds entry_bytes to _total_memory_bytes on every store without subtracting on overwrite, permanently inflating the counter and forcing spurious eviction. Because the module is unwired (0 importers), impact today is limited -- consolidate or delete.
- **Evidence (verbatim):**
```
if self.kv_data is not None and not self.children: return self.last_access, self ... for child in self.children.values(): t,n = child._find_lru_leaf()
```
- **Impact:** If ever wired, entries could never be LRU-evicted (leak) and memory budget would monotonically inflate; currently latent because the module is unused.
- **Effort:** 3-5 hours
- **Reliability:** RadixNode.evict_lru relies on _find_lru_leaf which returns leaves only; internal-node entries (kv_data + children) are never candidates. store() at line 233 increments _total_memory_bytes with no matching decrement on overwrite.
- **Recommendation:** If kept, make LRU traverse internal nodes too (find globally oldest last_access node regardless of leaf-ness) and make store() net-adjust _total_memory_bytes on overwrite. Prefer deleting in favor of dist/prefix_cache.PrefixCache and removing radix_tree_cache from the tree entirely.

---

### F-140 — [Low] GracefulDegradationHandler.degrade() resets _total_degradation_time_s to 0 on every event, zeroing recovery-time stats

`src/distllm/dist/backends/graceful_degradation.py:221` · zone=`backends-config-cloud` · category=`bug`

- **Summary:** graceful_degradation.py line 221 assigns self._total_degradation_time_s = 0.0 inside degrade() on every degradation event (comment even says 'tracked per-event'). stats() computes average_recovery_time_s from this accumulated total, which is repeatedly wiped, so the metric is always near zero and record_success accumulates into a reset counter. The circuit-breaker/fallback logic is unaffected, but the observability output is wrong.
- **Evidence (verbatim):**
```
self._total_degradation_time_s = 0.0  # tracked per-event
```
- **Impact:** average_recovery_time_s in stats() is meaningless; misleads operators about fallback recovery speed.
- **Effort:** 1 hour
- **Reliability:** degrade() runs on every fallback (called from any failure path), so _total_degradation_time_s is reset immediately before record_success later reads it (lines 221, 328-337).
- **Recommendation:** Remove the reset; accumulate into _current_degradation_start/_total_degradation_time_s only in record_success/record_failure, and compute avg_recovery_time_s from the per-event durations rather than a module-level total.

---

### F-141 — [Low] PartitionStore.compare_runs memory comparison always 0 — 'total_memory_gb' is never written by _solution_to_dict

`src/distllm/dist/partition/persistence.py:264` · zone=`dist-partition` · category=`bug`

- **Summary:** compare_runs reads run_a.solution.get('total_memory_gb',0)/run_b... but _solution_to_dict writes only PartitionSolution's own public fields (max_node_time_ms, points, estimated_throughput_tok_s, ...) and never a 'total_memory_gb' key. mem_a/mem_b are therefore always 0, and memory_diff_gb always 0.0 while the winner logic ignores memory anyway.
- **Evidence (verbatim):**
```
mem_a = run_a.solution.get("total_memory_gb", 0) mem_b = run_b.solution.get("total_memory_gb", 0) mem_diff = mem_b - mem_a   # always 0: key never serialized
```
- **Impact:** Memory comparison in the A/B run comparison is inert; users comparing runs cannot see memory deltas.
- **Effort:** 1-2 hours
- **Reliability:** Trace: _solution_to_dict (lines 373-392) serializes solution.__dict__ (PartitionSolution fields) with no 'total_memory_gb'; compare_runs therefore always sees 0.
- **Recommendation:** Serialize total_memory_gb (sum of point memory or cost_model.evaluate) in _solution_to_dict (or compute it in compare_runs from stored costs), and add a test for memory_diff_gb.

---

### F-142 — [Low] partition_model_across_nodes emits invalid (start>end) ranges when num_nodes > total_layers (and div-by-zero at 0)

`src/distllm/models/partition_planner.py:65` · zone=`core-training` · category=`bug`

- **Summary:** When node count exceeds layer count, layers_per_node=0 and non-first nodes get end = start-1 (inverted/negative ranges). num_nodes=0 raises ZeroDivisionError.
- **Evidence (verbatim):**
```
layers_per_node = total_layers // num_nodes ... end = start + layers_per_node + extra - 1 assignments.append((start, end))  # end can be < start
```
- **Impact:** Empty/inverted layer ranges for small models under many workers, causing downstream load/traversal errors.
- **Effort:** 1 hour
- **Reliability:** partition_model_across_nodes('x', 40) on a 32-layer model returns several (start, start-1) tuples.
- **Recommendation:** Clamp partitions to min(num_nodes, total_layers), validate num_nodes>0, and add tests for node-count > layer-count and zero.

---

### F-143 — [Low] Airflow/Kubeflow batch operators poll with no timeout - a stuck job blocks the DAG task forever

`src/distllm/plugins/airflow.py:92` · zone=`ops-utils` · category=`bug`

- **Summary:** DistLLMBatchOperator.execute and kubeflow_batch_inference_op loop `while status in ('running','pending','queued')` with a fixed 5s sleep and no attempt cap or overall deadline (airflow.py line 92, kubeflow.py line 71). If the coordinator reports running indefinitely (or the poll GET succeeds with a stale status), the Airflow DAG task / KFP component never completes.
- **Evidence (verbatim):**
```
while status in ("running", "pending", "queued"): time.sleep(5)  (lines 92-93)
```
- **Impact:** Hangs DAG runs, building up stuck task slots in Airflow orchestrations.
- **Effort:** 1-2 hours
- **Reliability:** Point endpoint at a stub that returns status='running' forever -> loop never exits.
- **Recommendation:** Add max_polls/timeout_seconds (with a configurable deadline) and raise/return a terminal status on expiry; surface a warning log with elapsed time.

---

### F-144 — [Low] verification/__init__.__all__ lists 'print_report' which does not exist; docs show await on a sync method

`src/distllm/verification/__init__.py:45` · zone=`ops-utils` · category=`bug`

- **Summary:** from distllm.verification import * raises AttributeError because __all__ includes 'print_report' (line 45) but no such symbol is defined anywhere in the package. Separately, runner.py's docstring/__init__ example uses 'report = await verifier.verify(...)' while verify() is a plain (non-async) method returning VerificationReport, so that snippet fails.
- **Evidence (verbatim):**
```
"print_report" in __all__ (line 45) with no definition in comparator/hash_registry/report/runner
```
- **Impact:** Star-imports of the public verification API break; copied usage sample fails immediately.
- **Effort:** 30 min - 1 hour
- **Reliability:** try `from distllm.verification import *` and `report = await verifier.verify(['x'])` -> AttributeError / TypeError.
- **Recommendation:** Remove 'print_report' from __all__ (or implement/export it), and either make verify() async or fix the docstring examples to omit await.

---

### F-145 — [Low] MultiDraftSpeculativeDecoder acceptance-rate stat can exceed 1.0 (correction tokens counted as acceptances)

`src\distllm\core\speculative_decoder.py:585` · zone=`core-decoding` · category=`bug`

- **Summary:** speculative_decoder.py MultiDraftSpeculativeDecoder.generate sets `self._stats['accepted'] = generated.shape[1] - prompt_len` (line 586), which counts EVERY non-prompt token including the correction token sampled fresh from the target on partial rejection (and single-token fallbacks when consensus_len==0). total_proposed is the sum of consensus lengths. So acceptance_rate = accepted/proposed can exceed 1.0 whenever single-token or correction tokens add tokens beyond what was proposed — exactly the bug distributed_speculative.py explicitly guards against with `min(accepted_delta, actual_draft_tokens)` (line 974).
- **Evidence (verbatim):**
```
self._stats['total_proposed'] = sum(self._stats['consensus_lengths']); self._stats['accepted'] = generated.shape[1] - prompt_len
```
- **Impact:** Dashboard/quality-scorer metrics overstate multi-draft acceptance, driving adaptive candidate-length and draft-model routing (DraftQualityScorer.record uses these) on inflated signals.
- **Reliability:** Repro path: a request where consensus_len==0 on the first iteration produces 1 target token with total_proposed=0 -> accepted=1, proposed=0 -> acceptance_rate guard max(total,1)=1 -> 1.0 from zero proposed; partial rejections inflate further.
- **Recommendation:** Accumulate accepted only as `accepted_count` at each iteration (like SpeculativeDecoder.generate lines 158-159 do with `self._stats['accepted'] += accepted_count`) and cap with min(accepted, total_proposed) before computing the rate, matching the distributed_speculative.py approach.

---

### F-146 — [Low] voyager ModalityEncoder.encode_audio passes a possible CUDA torch tensor into openai-whisper transcribe, which expects a numpy 16 kHz array

`src\distllm\core\voyager_multimodal.py:395` · zone=`core-gen-rag` · category=`bug`

- **Summary:** encode_audio moves audio to self._device (`audio_tensor = audio_tensor.to(self._device)`) and calls `model.transcribe(audio_tensor)`. openai-whisper's transcribe expects a np.float32 16 kHz audio array or a file path, not a CUDA tensor; on GPU this raises and is swallowed by the catch-all (line 406), so audio encoding silently degrades to the zero/sample-prefix fallback on every GPU request. Note transcribe is also called inside the thread pool without the encoder lock.
- **Evidence (verbatim):**
```
audio_tensor = audio_tensor.to(self._device) result = model.transcribe(audio_tensor) # openai-whisper expects a numpy float32 16kHz array or path
```
- **Impact:** Audio modality never produces a real embedding on CUDA/mps; upstream inference gets a meaningless audio embedding, undermining multimodal correctness.
- **Recommendation:** Pass the CPU float32 numpy array to transcribe (keep audio on CPU for whisper; it does its own device handling) and move decodes/features off a shared model, or acquire the encoder lock around the transcribe call; add an audio-ended test asserting a non-zero embedding on GPU.

---

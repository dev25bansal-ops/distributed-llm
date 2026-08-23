# DistLLM Threat Model

Status: initial version (2026-08-24). Every mitigation marked **Verified** was
confirmed by reading the cited source file; anything we could not confirm in
code is listed under [Open gaps](#open-gaps) instead of being claimed.

## 1. Scope and deployment topology

DistLLM pools GPUs across machines to run models no single device can handle.
The reference deployment has one **coordinator** (FastAPI API server + engine)
and N **worker nodes** over gRPC, with optional P2P gossip/DHT between peers,
optional cross-cluster federation, and an optional HA standby coordinator.

```
                       TB1: HTTP(S) + Bearer key / JWT / SSO
  Clients ──────────────────────────────────────► Coordinator (FastAPI)
                                                    │        ▲
                              TB2: gRPC (+cluster   │        │ TB4: HA heartbeat/
                              key, TLS optional)    ▼        │     snapshot (X-HA-Secret)
                                                 Worker ◄──────┘ Peer coordinator
                                                   ▲
                        TB3: P2P — gossip UDP (HMAC), DHT UDP
                        (HMAC store tokens), ICE/QUIC transport
                                                   │
                                        Other peer nodes (workers/coordinators)

  Coordinator ──► outbound fetches (HF hub, webhooks, eval providers)   [TB6]
  Cross-cluster federation (Ed25519-signed messages)                    [TB5]
```

## 2. Assets

| Asset | Where it lives | Impact if compromised |
|---|---|---|
| Model weights | worker memory/disk, HF cache | theft of proprietary/fine-tuned weights |
| Prompts & completions | request pipeline, KV cache, semantic cache | user data breach; prompt leakage |
| API keys | env/file → `ApiKeyStore` (salted hashes in memory) | impersonation of any tenant/role |
| Cluster key | shared secret for coordinator↔worker gRPC | full control of worker fleet |
| Gossip HMAC key (`DISTLLM_GOSSIP_HMAC_KEY` or `~/.distllm/gossip_hmac.key`) | all peers | forged cluster membership/cache-index updates |
| DHT shared secret | Kademlia STORE authorization | DHT poisoning |
| HA secret (`DISTLLM_HA_SECRET`) | leader/standby coordinators | state injection into a standby |
| Ed25519 identity keys | per-node signing keys (byzantine detection, federated merge, plugin manifests) | forged node messages / adapter submissions |
| TLS certificates & CA | `core/certificate_manager.py` rotation | MITM on gRPC if TLS enabled |

## 3. Trust boundaries

- **TB1 — Client ↔ Coordinator (HTTP/TLS):** untrusted Internet/LAN clients.
  Auth via Bearer API keys or JWT/SSO; health probes exempt.
- **TB2 — Coordinator ↔ Worker (gRPC):** semi-trusted cluster interior.
  Shared cluster key per RPC; TLS optional.
- **TB3 — Worker ↔ Worker P2P:** peers are mutually distrustful.
  Gossip HMAC, DHT store-capability tokens, ICE MESSAGE-INTEGRITY.
- **TB4 — Coordinator ↔ Peer coordinator (HA):** heartbeat is auth-exempt but
  gated by `X-HA-Secret`; snapshot endpoints fail closed without it.
- **TB5 — Federation (cross-cluster):** different administrative domains;
  Ed25519-signed messages with registered peer public keys.
- **TB6 — Coordinator ↔ External services:** model downloads, webhooks,
  eval/provider callbacks. SSRF guards apply.

## 4. STRIDE analysis

### Spoofing

| Threat | Mitigation | Status |
|---|---|---|
| Forged/stolen client identity | Bearer keys validated against PBKDF2-HMAC-SHA256 hashes (100k iterations, 16-byte per-key salt, constant-time compare); role attached to `request.state` | Verified — `src/distllm/core/api_key_store.py`, `src/distllm/api/middleware.py` |
| JWT algorithm confusion / weak-HS256 fallback | PyJWT required as primary validator; pure-Python fallback defense-in-depth only | Verified by tests — `tests/security/test_jwt_algorithm_confusion.py`, `test_jwt_hs256_fallback.py`; fallback fix applied 2026-08-11 |
| SSO/OAuth replay & CSRF | state parameter + OIDC nonce | Present — `src/distllm/api/sso_auth.py`; tests `test_oauth_state_csrf.py`. Not re-audited this pass |
| Rogue worker joining gRPC cluster | every RPC carries `cluster_key`; servicer rejects when unset (**fail closed**) and compares constant-time | Verified — `src/distllm/dist/node_service.py:131-146` |
| Forged gossip messages | HMAC-SHA256 over canonical serialization; missing signature rejected; construction **requires** a key (env var or persisted file), else raises | Verified — `src/distllm/dist/p2p/gossip.py:236-266, 305-326` |
| DHT STORE poisoning | time-bound HMAC capability token binding sender+key+value+expiry (TTL 300 s, skew 300 s, constant-time verify) | Verified with caveat — `src/distllm/dist/p2p/kademlia_dht.py:865-933`. Fails open to unauthenticated mode if no secret configured (see gaps) |
| Forged federation messages / adapter submissions | Ed25519 signatures from registered per-node public keys; unsigned or invalid submissions rejected (fail closed), signature binds `node_id` + round | Verified — `src/distllm/dist/byzantine.py:49-930`, `src/distllm/dist/federated_merge.py:127-228` |

### Tampering

| Threat | Mitigation | Status |
|---|---|---|
| In-transit message tampering (P2P) | HMAC covers full canonical body of gossip/DHT payloads | Verified — files above |
| KV-cache poisoning across nodes | KV integrity checks + Merkle digests | Present — `src/distllm/dist/merkle.py`, `cache_digest.py`; tests `tests/security/test_kv_cache_integrity.py`, `test_cache_poisoning.py`. Mechanism not re-read line-by-line this pass |
| Malicious model/plugin supply chain | HF revision pinning helper (`DISTLLM_MODEL_REVISION`, hard-require mode); plugin manifests signed with Ed25519 and verified before load | Verified for helpers — `src/distllm/security/utils.py:14-36`, `src/distllm/core/plugin_sandbox.py`. Sandbox runtime isolation not audited this pass |
| Cache-entry tampering across tenants | tenant-scoped cache keys | Fixed + adversarially verified 2026-08-11; regression test `tests/security/test_cache_plugin_tenant_isolation.py` |
| Dedup-cache poisoning | `DedupMiddleware` exists; internal audit still lists a dedup authentication bypass as an **open release blocker** | Open — see gaps |

### Repudiation

| Threat | Mitigation | Status |
|---|---|---|
| "We never sent that prompt" disputes | Request-ID middleware on every response; prompt-injection audit log setting (`DISTLLM_INJECTION_AUDIT_LOG`) | Partially verified — middleware wired at `server.py:687`; audit log is local-file based |
| Tampered/PII-leaking logs | Log-redaction module (`LogRedactor`, `RedactingFilter`) | **Implemented but not wired** — `install_global_redaction()` is never called at startup (see gaps) |

### Information disclosure

| Threat | Mitigation | Status |
|---|---|---|
| Prompt/completion interception between workers | E2E encryption: X25519 key exchange + XSalsa20-Poly1305 AEAD, signed key exchange, AAD-bound session keys; tensor-payload helpers | Verified module — `src/distllm/security/e2e.py`; wired into gRPC servicer as opt-in constructor arg (`node_service.py:122-126`) |
| SSRF into private network (webhooks, eval URLs, provider URLs) | `validate_http_url` blocks private/loopback/link-local/multicast/reserved IPs; `safe_urlopen` resolves DNS once and connects to the IP (rebinding protection) while preserving Host header | Verified — `src/distllm/security/utils.py`; regression tests `test_ssrf_bypass.py`, `test_ssrf_federation.py`, `test_dify_provider_urls.py` |
| Secrets/keys in logs | redaction module (unwired — see gaps); auth fingerprint explicitly redacted in general error context | Gap + partial (`server.py:492`) |
| PII in stored prompts | Content-moderation pipeline incl. `PIIRedactor` wired as middleware | Wired — `server.py:708`; detectors in `src/distllm/security/content_moderation/` |
| Eavesdropping on coordinator↔worker link | TLS supported end-to-end (client creds, server creds, cert rotation) but **off by default** | Gap — see below |

### Denial of Service

| Threat | Mitigation | Status |
|---|---|---|
| API-key brute force | per-IP failed-auth rate limiter escalates to 429 lockout | Verified — `src/distllm/api/middleware.py:287-329` |
| Request floods | unified request rate limiter + backpressure + circuit breaker middlewares | Wired — `server.py:692, 900, 906` |
| Oversized payloads | request size limit middleware; gRPC message caps; protobuf input-size validation (batch ≤1024, hidden dim ≤16384, seq ≤131072, layers ≤256) | Verified — `server.py:809`, `node_service.py:116-120, 155-158` |
| Slow-loris / hung handlers | timeout middleware | Wired — `server.py:662` |
| Unauthenticated CPU burn via injection scanner | none effective — scanner runs outermost relative to auth (repo's own audit flags ordering) | Open gap |
| DHT/gossip floods | HMAC requirement blocks unauthenticated senders *when secrets are configured* | Conditional |

### Elevation of Privilege

| Threat | Mitigation | Status |
|---|---|---|
| Keyholder exceeding their role | six-role hierarchy (`admin` … `read-only`) enforced by `require_role()` which fails closed (401 when no role attached, 403 on insufficient role) | Verified — `src/distllm/api/auth_deps.py`, `api_key_store.py:55-87` |
| Auth bypass via env vars (`DISTLLM_NO_AUTH`, `DISABLE_AUTH`) | removed; middleware logs CRITICAL and still enforces auth | Verified — `middleware.py:215-231` |
| Unauthenticated admin/plugins access | admin-gated plugins; DocsAuthMiddleware on docs/dashboard routes | Wired — `server.py:738`; plugin gating per Focus-Areas hardening round |
| IDOR on exchange routes | object-ownership checks | Regression-tested — `tests/security/test_idor_exchange.py` |
| State injection into HA standby | `/api/v1/ha/*` fail **closed** when `DISTLLM_HA_SECRET` unset (403), constant-time header compare otherwise | Verified — `server.py:1403-1423` |
| Code escape via plugins | plugin sandbox (manifest + Ed25519 signature verification) | Signature path verified; sandbox enforcement depth not audited this pass |
| TEE/edge attestation spoofing | attestation scaffolding exists | **Dev scaffold only** — `security/tee.py` self-labels the dev key as SCAFFOLD; not production attestation |

## 5. CI security gates

| Gate | Workflow | Mode |
|---|---|---|
| Bandit SAST | `.github/workflows/ci.yml` (sast job) | enforcing |
| Secret detection | ci.yml (detect-secrets vs baseline) | enforcing vs baseline |
| Dependency vulnerabilities | `.github/workflows/dependency-scan.yml` (pip-audit over installed project deps) | **advisory** (`continue-on-error: true`) until findings triaged — TODO to flip to enforcing |
| Security regression suite | `tests/security/**` (~27 files incl. prompt-injection bypass, SSRF bypass, rate-limit bypass, pickle ban) | enforcing in CI test job |

## 6. Open gaps

Honest list of things this threat model cannot claim today:

1. **gRPC links are plaintext by default.** `create_node_client` /
   `create_async_node_client` take `use_tls=False` as default
   (`src/distllm/dist/node_client.py`). With TLS off, the cluster key rides
   inside the cleartext channel, so a passive on-path observer can capture it.
   Enable `use_tls` (or front with mTLS via `core/certificate_manager.py`)
   on any untrusted network.
2. **API server ships no TLS termination.** All launch paths call
   `uvicorn.run(...)` without SSL options (`cli/main.py:1325`,
   `cli/system_commands.py:161`). HTTPS depends on an external reverse proxy —
   currently an undocumented operational requirement.
3. **DHT fails open without configuration.** An empty `shared_secret`
   silently degrades Kademlia STORE to unauthenticated mode
   (`kademlia_dht.py:876-882`) after a loud warning. Prefer refusing to start
   in P2P deployments where DHT is enabled.
4. **Log redaction is implemented but never installed.** No startup path calls
   `install_global_redaction()` (`security/log_redaction.py:197`), so prompts/
   PII can reach logs unredacted.
5. **WAF is implemented but not attached.** `WAFMiddleware`
   (`src/distllm/api/waf.py`) has no default registration in `server.py`;
   deployments must call `add_waf_middleware()` themselves.
6. **Middleware ordering exposes pre-auth scanning.** `PromptInjectionMiddleware`
   is registered after `AuthMiddleware` in `server.py` (lines 682 vs 698),
   which makes it run *before* auth in Starlette's wrapping order —
   unauthenticated requests consume classifier CPU. The repo's own
   `api/strategic_report.md` tracks this.
7. **Known open release blockers from the internal audit** (2026-08-11):
   dedup-middleware authentication bypass and sibling-cache token issues
   remain unfixed at time of writing.
8. **Single shared cluster key.** Any single compromised worker can impersonate
   the coordinator toward every other worker; there is no per-node gRPC
   identity unless TLS client certs are deployed.
9. **TEE attestation is a dev scaffold** (`security/tee.py`); do not rely on
   it for hardware-rooted guarantees yet.
10. **Dependency gate is advisory.** Flip `continue-on-error` off in
    `.github/workflows/dependency-scan.yml` once current findings are triaged.

## 7. Verification method

Claims were checked against source on 2026-08-24 by reading:
`core/api_key_store.py`, `api/middleware.py`, `api/auth_deps.py`,
`api/prompt_injection.py`, `api/server.py` (middleware registration +
HA secret rejection), `dist/node_service.py`, `dist/node_client.py`,
`dist/p2p/gossip.py`, `dist/p2p/kademlia_dht.py`, `dist/byzantine.py`,
`dist/federated_merge.py`, `security/utils.py`, `security/e2e.py`,
`security/log_redaction.py`, `security/waf` wiring, and the
`tests/security/` index. Items marked "Present" exist and have regression
tests but were not line-by-line re-verified in this pass.

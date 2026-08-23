---
tags:
  - security
  - compliance
  - trust
  - moderation
aliases:
  - Security
  - Security & Compliance
---
# Security & Compliance — `src/distllm/{security,compliance}/` + auth surfaces

**18 .py files · ~4.3K LOC** in `security/`+`compliance/`, **plus** the pervasive auth/SSO/WAF/moderation surfaces in [[03 API Server]] and the key-rotation/secret/compliance machinery in [[01 Core Engine]].

> The trust & safety wall: **end-to-end encryption** of tensor bytes between nodes (X25519 + XSalsa20-Poly1305 with key ratchet), **SSRF/DNS-rebinding-safe HTTP**, **model watermarking**, **content moderation** (toxicity / PII / jailbreak / topic), attestation scaffolds (SPIFFE, TEE, edge), and **auditor-facing compliance evidence packs**.

## `security/`

| file | LOC | purpose |
|------|-----|---------|
| `e2e.py` | 358 | `E2EEncryption`/`SessionKeys` — NaCl X25519 + XSalsa20-Poly1305 per-session, key ratchet forward secrecy |
| `edge_attestation.py` | 325 | **SCAFFOLD** mTLS + device-attestation gate (`PLUGIN:` points for TPM/Nitro/vTPM) |
| `log_redaction.py` | 58 | `LogRedactor` — regex strip PII/tokens |
| `quantum_safe_tls.py` | 258 | post-quantum TLS scaffold (opt-in ML-KEM hybrid + ALPN signal) |
| `spiffe.py` | 292 | SPIFFE/SVID zero-trust scaffold, dev CA (`DEV ONLY`) |
| `tee.py` | 387 | **SCAFFOLD** software TEE/confidential-computing sim (enclave, seal/unseal, attestation) |
| `utils.py` | 117 | `hf_revision`, `validate_http_url`, `safe_urlopen` (DNS-rebinding / SSRF hardened) |
| `watermark.py` | 626 | `ModelWatermark`/`WeightWatermark`/`GumbelWatermark` + CLI |
| `content_moderation/__init__.py` | 39 | re-exports 5 detectors |
| `content_moderation/base.py` | 333 | result types + backend ABC (transformers/ONNX/keyword) |
| `content_moderation/toxicity.py` | 166 | `ToxicityDetector` |
| `content_moderation/pii.py` | 227 | `PIIRedactor` (regex + spacy optional) |
| `content_moderation/jailbreak.py` | 192 | `JailbreakDetector` |
| `content_moderation/topics.py` | 231 | `TopicFilter` allow/deny policies |
| `content_moderation/pipeline.py` | 125 | `ContentModerationPipeline` — orchestrates 4 detectors |
| `__init__.py` | 60 | re-exports public API |

## `compliance/`

| file | LOC | purpose |
|------|-----|---------|
| `evidence_pack.py` | 618 | `EvidencePack`/`ControlRef`/`build_evidence_pack()` — folds core `compliance_evidence` + GDPR/EXPORT/HIPAA parsing, JSON/Markdown emitters |
| `__init__.py` | 8 | shim |

## Related auth & enforcement surfaces (cross-referenced)

- **[[03 API Server]]** — `AuthMiddleware` (Bearer/JWT/X-API-Key), `require_role` RBAC, SSO (`auth/{saml,oidc,oauth2}`), OPA/Rego `authz/opa`, `WAF`, CSRF, rate-limit, quota, prompt-injection (`prompt_injection.py`).
- **[[01 Core Engine]]** — `api_key_store` (Argon2), `certificate_manager`/`cert_rotation`, `secret_manager`, `request_auditor`, `aegis_compliance`, `compliance_evidence`, differential privacy, `plugin_sandbox`.
- **[[11 Platform Services]]** — `verification/hash_registry`, `security` reuse.

## Notes / dead code
- `edge_attestation.py`, `spiffe.py`, `tee.py` are explicit **software scaffolds** with dev keys on disk (`DEV ONLY`).
- `security/*content_moderation` re-exported; async `thread-pool` variants provided per detector.

## Tests
`tests/security/` (~23 files): JWT/auth, CSRF, SSRF, cache poisoning, content moderation, log redaction, pickle ban, idor, input validation, vulnerabilities/comprehensive sweeps. Plus `tests/security_pkg/`, `tests/edge/`.
# Security Hardening Guide

<!-- TOC -->
## Table of Contents

- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [TLS Configuration](#tls-configuration)
  - [Generate Certificates](#generate-certificates)
  - [Enable TLS](#enable-tls)
  - [Verify TLS](#verify-tls)
- [Authentication](#authentication)
  - [API Key Management](#api-key-management)
  - [Cluster Key (gRPC Authentication)](#cluster-key-grpc-authentication)
  - [RBAC Roles](#rbac-roles)
- [Network Security](#network-security)
  - [Firewall Rules](#firewall-rules)
  - [CORS Configuration](#cors-configuration)
  - [Disable Public Docs](#disable-public-docs)
- [Secret Management](#secret-management)
  - [Environment Variables (Development)](#environment-variables-development)
  - [HashiCorp Vault (Production)](#hashicorp-vault-production)
  - [AWS Secrets Manager](#aws-secrets-manager)
- [Input Validation](#input-validation)
  - [SSRF Protection](#ssrf-protection)
  - [Request Size Limits](#request-size-limits)
  - [Rate Limiting](#rate-limiting)
- [Container Security](#container-security)
  - [Non-Root User](#non-root-user)
  - [Read-Only Filesystem](#read-only-filesystem)
  - [Security Context (Kubernetes)](#security-context-kubernetes)
- [Monitoring & Alerting](#monitoring--alerting)
  - [Log Redaction](#log-redaction)
  - [Security Metrics](#security-metrics)
  - [Alerting Rules](#alerting-rules)
- [Debug Endpoints](#debug-endpoints)
- [Incident Response](#incident-response)
  - [If API Key is Compromised](#if-api-key-is-compromised)
  - [If Node is Compromised](#if-node-is-compromised)
  - [If Coordinator is Compromised](#if-coordinator-is-compromised)
- [Compliance](#compliance)
  - [GDPR](#gdpr)
  - [SOC 2](#soc-2)
  - [HIPAA](#hipaa)
<!-- /TOC -->

## Pre-Deployment Checklist

- [ ] Enable TLS for all connections
- [ ] Set strong API keys (not default/empty)
- [ ] Configure firewall rules
- [ ] Enable authentication on all endpoints
- [ ] Set up RBAC for different user roles
- [ ] Enable audit logging
- [ ] Configure CORS for your domains only
- [ ] Disable OpenAPI docs in production
- [ ] Set up secret management (not env vars in production)
- [ ] Enable rate limiting

---

## TLS Configuration

### Generate Certificates

```bash
# Self-signed (development only)
distllm security cert create --hostname my-cluster.example.com

# Let's Encrypt (production)
certbot certonly --standalone -d my-cluster.example.com
```

### Enable TLS

```yaml
# config.yaml
tls:
  enabled: true
  cert_file: /etc/distllm/tls/cert.pem
  key_file: /etc/distllm/tls/key.pem
  ca_cert_file: /etc/distllm/tls/ca.pem
```

### Verify TLS

```bash
# Test TLS connection
openssl s_client -connect my-cluster:50050 -servername my-cluster

# Verify certificate
openssl x509 -in cert.pem -text -noout
```

---

## Authentication

### API Key Management

```bash
# Set API key via environment variable
export API_KEY=your-secure-key-here

# Start server with API key
distllm system api --model <model> --local

# Or disable auth for development
distllm system api --model <model> --local --no-auth
```

The API key is displayed on startup. For production, use a strong random key:

```bash
export API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Cluster Key (gRPC Authentication)

```bash
# Set cluster key for node-to-node authentication
export DISTLLM_CLUSTER_KEY=your-cluster-secret

# All nodes must use the same key
```

### RBAC Roles

| Role | Permissions |
|------|-------------|
| `admin` | Full access (create, delete, manage) |
| `inference` | Generate completions only |
| `read-only` | View status, models, metrics |
| `marketplace` | Create/manage marketplace listings |

```python
# Set role for API key
from distllm.core.api_key_store import ApiKeyStore
store = ApiKeyStore()
store.add_key(key_str="...", role="inference-only", label="my-key")  # returns key_id
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — this snippet previously showed
> `store.create_key(name=..., role=...)`; the real method is `add_key(key_str,
> role, label, key_id)` per `src/distllm/core/api_key_store.py:320`, and valid
> roles are admin/model-admin/user-admin/inference-only/read-only/auditor
> (`VALID_ROLES`, line 68) — `inference` is not a role.

---

## Network Security

### Firewall Rules

```bash
# Allow only necessary ports
ufw allow 8000/tcp   # API
ufw allow 50050/tcp  # gRPC (coordinator)
ufw allow 50051/tcp  # gRPC (workers)
ufw allow 9091/tcp   # Metrics (internal only)
ufw deny 3000/tcp    # Grafana (use SSH tunnel)
```

### CORS Configuration

```yaml
# config.yaml
coordinator:
  cors_origins: "https://your-app.com,https://admin.your-app.com"
```

### Disable Public Docs

```bash
# Disable OpenAPI/Swagger in production
export DISTLLM_ENABLE_DOCS=false
```

---

## Secret Management

### Environment Variables (Development)

```bash
export DISTLLM_API_KEY=sk-...
export DISTLLM_CLUSTER_KEY=...
export HUGGING_FACE_HUB_TOKEN=hf_...
```

### HashiCorp Vault (Production)

```python
from distllm.core.secret_manager import SecretManager

mgr = SecretManager(backend="vault", url="https://vault:8200", token="...")
api_key = mgr.get_secret("distllm/api-key")
```

### AWS Secrets Manager

```python
mgr = SecretManager(backend="aws", region="us-east-1")
api_key = mgr.get_secret("distllm/api-key")
```

---

## Input Validation

### SSRF Protection

SSRF protection is always enabled for image URLs in chat requests. The system:
- Validates DNS resolution
- Blocks private IP ranges (10.x, 172.16.x, 192.168.x, 127.x)
- Prevents DNS rebinding attacks

### Request Size Limits

```yaml
# config.yaml
api:
  max_request_size_mb: 32
  max_message_length: 131072
```

### Rate Limiting

```yaml
# config.yaml
api:
  rate_limit_requests: 1000
  rate_limit_window_seconds: 60
```

---

## Container Security

### Non-Root User

```dockerfile
# Already configured in Dockerfile
USER distllm
```

### Read-Only Filesystem

```yaml
# docker-compose.yml
services:
  coordinator:
    read_only: true
    tmpfs:
      - /tmp
      - /var/cache/distllm
```

### Security Context (Kubernetes)

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault
```

---

## Monitoring & Alerting

### Log Redaction

All application logs are automatically redacted at startup to prevent secrets and PII leakage. `install_global_redaction()` is called in `api/server.py::_init_observability`, which runs from the FastAPI `lifespan` handler (`app = FastAPI(lifespan=lifespan, ...)`) — the live startup path. It installs two layers:

- **Loguru core patcher**: applied by loguru to every `loguru.logger` record (the primary logging path across the codebase) after %-formatting and before any sink writes — covers stderr, JSON stdout, `enqueue=True` queues, and third-party sinks. No call site needs to change.
- **Stdlib `RedactingFilter`**: attached to the root logger and each of its handlers, covering libraries that log via stdlib logging (including records propagated from child/third-party loggers).

The pattern set lives in `src/distllm/security/log_redaction.py` (`DEFAULT_PATTERNS`):

| Category | Patterns |
|----------|----------|
| **PII** | email addresses, SSN, phone numbers, credit card numbers, IP addresses |
| **API keys** | `sk-...` tokens, `api_key="..."` assignments, AWS access keys (`AKIA...`) |
| **Credential assignments** | password/passwd/pwd, secret, client_secret, token, access_token, refresh_token `= value` |
| **Authorization headers** | `Bearer <token>`, Authorization / Proxy-Authorization header values |
| **High-entropy tokens** | base64/base64url runs ≥40 chars (covers `ApiKeyStore` `secrets.token_urlsafe(48)` keys with no prefix), hex runs ≥32 chars |

Redacted spans are replaced with `[REDACTED]`.

Configuration:
- `DISTLLM_LOG_REDACTION=off` (also `0`, `false`, `no`) disables redaction (checked at emit time as well as install time)
- `DISTLLM_REDACT_PATTERNS` — extra semicolon-separated regexes layered on top of the defaults

Residual gap: only `record["message"]` is scrubbed on the loguru path — exception traceback text (`record["exception"]`) is not rewritten, so a secret embedded only in an exception repr can still appear in `logger.exception()` traceback output. This is acknowledged in `tests/security/test_loguru_redaction.py::test_exception_logging_redacted`.

**Test coverage:** `tests/security/test_loguru_redaction.py::TestInstallGlobalRedactionCombined::test_startup_wiring_call_sites_exist` asserts that `_init_observability` in `distllm/api/server.py` (and the mirrored copy in `distllm/api/server_lifespan.py`, which currently has no importer) still contains the `install_global_redaction()` call, guarding against the "module tested, wiring untested" regression.

### Security Metrics

```bash
# Auth failures
distllm_auth_failures_total  # ⚠️ UNVERIFIED (hygiene pass 2026-11-06): no distllm_metrics_* metric exists; real names are distllm_* per src/distllm/observability/exporter.py — and no auth/rate-limit/SSRF metric is among the 25 defined there

# Rate limit hits
distllm_rate_limit_hits_total  # ⚠️ UNVERIFIED (hygiene pass 2026-11-06): same as above — name pattern corrected, metric not in exporter.py

# SSRF attempts blocked
distllm_ssrf_blocked_total  # ⚠️ UNVERIFIED (hygiene pass 2026-11-06): same as above — name pattern corrected, metric not in exporter.py
```

### Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: distllm-security
    rules:
      - alert: HighAuthFailureRate
        expr: rate(distllm_auth_failures_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High authentication failure rate"
```

---

## Debug Endpoints

The `/v1/debug/*` routes provide debugging/replay capabilities. They are **not mounted on the live server** (`src/distllm/api/server.py` lists `debug_router` under `INTENTIONAL_UNMOUNT` at line 1116) and default to disabled even when mounted.

**Routes provided by `src/distllm/api/routes/debug.py`:**

| Route | Method | Purpose |
|-------|--------|---------|
| `/v1/debug/recent` | GET | List last N requests from replay buffer (truncated prompts) |
| `/v1/debug/request/{id}` | GET | Full details of a single stored request |
| `/v1/debug/replay` | POST | Replay a stored request through the model |
| `/v1/debug/deterministic` | POST | Enable/disable deterministic generation mode |
| `/v1/debug/buffer/export` | GET | Export up to 200 entries from replay buffer (truncated fields) |

**Access controls (layered — all must pass when the router is mounted):**

1. **RBAC**: The router is constructed with `APIRouter(dependencies=[Depends(require_role("admin"))])`, so every endpoint requires an identity with the `admin` role. `require_role` fails closed (401 when no role is attached, 403 on insufficient role) — `src/distllm/api/auth_deps.py`.

2. **Environment gate**: `DISTLLM_ENABLE_DEBUG_ROUTES` must equal `"1"`; otherwise every endpoint returns 404 (indistinguishable from an unmounted route).

3. **Admin key requirement**: `DISTLLM_ADMIN_KEY` must be set; if unset, requests return 503 ("Debug admin API not configured").

4. **Bearer token verification**: The Authorization header must carry `Bearer <key>` matching `DISTLLM_ADMIN_KEY` via `hmac.compare_digest` (constant-time).

**Audit logging:** Replay requests are recorded via `_audit_replay()` — actor key ID, role, request ID, client IP, and path are logged before the replay executes (`routes/debug.py`).

**Security considerations:**

- These routes expose full prompt/response data (including a detail endpoint that returns whole prompts) and should never be mounted in production
- Replaying requests can re-trigger expensive model inference — monitor usage
- Exported buffers may contain sensitive data — treat exports as confidential

**Verification:** Access-control structure verified by reading `src/distllm/api/routes/debug.py` (2026-11-05); there is currently no dedicated live-server test for these routes (the section in `tests/api/test_api_routes_comprehensive.py` is inside a module that is `pytest.skip`ped at import).

---

## Incident Response

### If API Key is Compromised

1. Immediately rotate the key: `distllm config setup`
2. Update all clients with new key
3. Review access logs for unauthorized usage
4. Check for data exfiltration

### If Node is Compromised

1. Isolate the node: `distllm cluster drain <node-id>`
2. Redistribute layers to other nodes
3. Investigate the compromised node
4. Rotate cluster key if necessary

### If Coordinator is Compromised

1. Stop the coordinator
2. Restore from backup
3. Rotate all API keys and cluster keys
4. Restart with fresh state

---

## Compliance

### GDPR

- Enable data residency: `DISTLLM_DATA_RESIDENCY=eu`
- Enable audit logging: `DISTLLM_AUDIT_LOGGING=true`
- Configure data retention: `DISTLLM_DATA_RETENTION_DAYS=30`

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06): none of these environment
> variables exists in `src/` (0 hits each for DISTLLM_DATA_RESIDENCY,
> DISTLLM_AUDIT_LOGGING, DISTLLM_DATA_RETENTION_DAYS). These are aspirational
> flags, not shipped config; delete the key names if a source of truth is
> needed.

### SOC 2

- Enable audit logging
- Configure access controls (RBAC)
- Enable encryption at rest and in transit
- Set up monitoring and alerting

### HIPAA

- Enable encryption for all data
- Configure access controls
- Enable audit logging
- Sign BAA with cloud provider

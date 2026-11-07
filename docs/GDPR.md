# GDPR Compliance

Data handling practices for DistLLM in compliance with the General Data Protection Regulation (EU) 2016/679.

---

## Data Processing Overview

### What Data Does DistLLM Process?

| Data Type | Purpose | Retention | Encrypted |
|-----------|---------|-----------|-----------|
| **Prompts** (user input) | Inference | In-memory only, not persisted by default | In transit (TLS) |
| **Generated text** (model output) | Response delivery | In-memory only | In transit (TLS) |
| **API keys** | Authentication | Hashed in storage | At rest |
| **Request metadata** | Rate limiting, billing | Configurable (default 30 days) | At rest |
| **KV cache** | Inference optimization | GPU memory, evicted on LRU | N/A (in-memory) |
| **Logs** | Operations, debugging | Configurable retention | At rest |
| **Audit logs** | Security compliance | Configurable (default 90 days) | At rest |

### What Data Does DistLLM NOT Store?

- Prompts are **not** persisted to disk by default
- Generated text is **not** persisted to disk by default
- No user profiles or behavioral tracking
- No data is sent to external services (unless explicitly configured)

---

## Data Processing Principles

### 1. Purpose Limitation

Data is processed solely for the purpose of providing LLM inference services:
- Prompts are processed to generate responses
- API keys are processed for authentication
- Request metadata is processed for rate limiting and billing

### 2. Data Minimization

- Only the minimum necessary data is collected
- Prompts can be configured to not appear in logs
- KV cache entries are automatically evicted when memory is full

### 3. Storage Limitation

- Request logs: configurable retention (default 30 days)
- Audit logs: configurable retention (default 90 days)
- KV cache: in-memory only, evicted by LRU
- No permanent storage of prompts or responses

### 4. Integrity and Confidentiality

- TLS encryption for all data in transit (configurable)
- Encryption at rest for persistent stores (configurable)
- API key hashing with SHA-256
- E2E encryption for inter-node communication (optional)

---

## Federated Inference Data Handling

When using distributed/federated inference:

### Data Flow

```
User → API Server → Coordinator → Worker Nodes → Model
                                     ↓
                              KV Cache (GPU memory)
```

### Data at Each Stage

| Stage | Data | Location | Duration |
|-------|------|----------|----------|
| API Server | Prompt text | Memory | Request lifetime |
| Coordinator | Request metadata | Memory | Request lifetime |
| Worker Node | Prompt tokens, KV cache | GPU memory | Until eviction |
| KV Cache | Hidden states | GPU memory | Until LRU eviction |

### Cross-Node Data Transfer

- Data transferred between nodes via gRPC with TLS
- KV cache transfers use the same encrypted channel
- No data leaves the cluster boundary

### Federated Fine-Tuning

When using federated fine-tuning:
- **Only gradient updates** are shared between nodes (not raw data)
- Local training data **never leaves** the node
- Differential privacy adds noise to gradients before sharing
- Secure aggregation (SecAgg) prevents any node from seeing individual gradients

---

## Data Subject Rights

### Right of Access

No per-user data-subject endpoint exists yet (`grep -rn '/v1/requests' src/distllm/api` returns 0 hits). Request your data via the support contacts below; the closest existing per-user read APIs are the prompt-exchange library (`GET /v1/exchange/library/{user_id}`, `src/distllm/api/routes/exchange.py:336`) and wallet (`GET /v1/exchange/wallet/{user_id}`, line 305) listings. See also `docs/api.md`.

### Right to Erasure

No per-user data-deletion endpoint exists yet (`grep -rn '/v1/requests' src/distllm/api` returns 0 hits). Submit an erasure request via the support contacts below.

### Right to Data Portability

No per-user data-export endpoint exists yet (`grep -rn '/v1/requests' src/distllm/api` returns 0 hits). Request a JSON export via the support contacts below; the closest existing per-user read APIs are `GET /v1/exchange/library/{user_id}` and `GET /v1/exchange/wallet/{user_id}` (`src/distllm/api/routes/exchange.py`). See also `docs/api.md`.

---

## Configuration for GDPR Compliance

### Disable Prompt Logging

No prompt-logging config keys exist yet (`grep -rn 'audit_log_prompts' src/distllm/config` returns 0 hits). Prompts are not persisted to disk by default (see "What Data Does DistLLM NOT Store?" above).

### Enable TLS

```yaml
tls:
  enabled: true
  cert_file: "/path/to/cert.pem"
  key_file: "/path/to/key.pem"
```

### Configure Data Retention

No retention/audit config keys exist yet (`grep -rn 'retention_days' src/distllm/config` returns 0 hits). Retention is operational (log rotation / storage lifecycle), not configurable through `config.yaml`.

### Enable E2E Encryption

```yaml
coordinator:
  cluster_key: "your-strong-cluster-key"
```

(`coordinator.cluster_key`, `src/distllm/core/coordinator_config.py:61`; env var `DISTLLM_CLUSTER_KEY`, read in `src/distllm/cli/cluster.py:317` and `src/distllm/cli/main.py:577`. There is no `security.e2e_encryption` key — 0 hits in `src/distllm/config`.)

---

## Audit and Compliance

### Audit Logs

All API access is logged with:
- Timestamp
- Client IP (anonymized option available)
- API key ID (not the key itself)
- Endpoint accessed
- Response status code
- Token counts

### Compliance Checklist

- [ ] TLS enabled for all connections
- [ ] API keys stored hashed (SHA-256)
- [ ] Prompt logging disabled in production
- [ ] Data retention policies configured
- [ ] E2E encryption enabled for federated inference
- [ ] Audit logs enabled with appropriate retention
- [ ] Access controls configured (RBAC)
- [ ] Backup encryption enabled

---

## Contact

For GDPR-related inquiries:
- Email: privacy@distllm.dev
- DPO: dpo@distllm.dev

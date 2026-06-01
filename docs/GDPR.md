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

Users can query their request history via the API:
```bash
curl -H "Authorization: Bearer $KEY" http://localhost:8000/v1/requests?user_id=USER_ID
```

### Right to Erasure

Users can request deletion of their data:
```bash
curl -X DELETE -H "Authorization: Bearer $KEY" http://localhost:8000/v1/requests?user_id=USER_ID
```

### Right to Data Portability

Request history can be exported in JSON format:
```bash
curl -H "Authorization: Bearer $KEY" http://localhost:8000/v1/requests?user_id=USER_ID&format=json
```

---

## Configuration for GDPR Compliance

### Disable Prompt Logging

```yaml
# config.yaml
coordinator:
  audit_log_prompts: false  # Don't log prompt content
  audit_log_responses: false  # Don't log response content
```

### Enable TLS

```yaml
tls:
  enabled: true
  cert_file: "/path/to/cert.pem"
  key_file: "/path/to/key.pem"
```

### Configure Data Retention

```yaml
logging:
  retention_days: 30
audit:
  retention_days: 90
```

### Enable E2E Encryption

```yaml
security:
  e2e_encryption: true
  cluster_key: "your-strong-cluster-key"
```

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

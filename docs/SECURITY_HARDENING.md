# Security Hardening Guide

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
store.create_key(name="my-key", role="inference")
```

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

### Security Metrics

```bash
# Auth failures
distllm_metrics_auth_failures_total

# Rate limit hits
distllm_metrics_rate_limit_hits_total

# SSRF attempts blocked
distllm_metrics_ssrf_blocked_total
```

### Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: distllm-security
    rules:
      - alert: HighAuthFailureRate
        expr: rate(distllm_metrics_auth_failures_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High authentication failure rate"
```

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

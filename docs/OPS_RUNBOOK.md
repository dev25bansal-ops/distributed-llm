# Operations Runbook

Standard operating procedures for DistLLM cluster operations.

---

## Rollback Procedures

### Rolling Back a Model Deployment

**Scenario**: New model version causes quality degradation or errors.

**Steps**:
```bash
# 1. Check current model version
distllm model list --json

# 2. Rollback to previous version (if using model version manager)
curl -X POST http://localhost:8000/admin/v1/models/rollback

# 3. Or load a specific known-good model
distllm deploy --hf meta-llama/Llama-2-70b --nodes 4 --wait

# 4. Verify health
curl http://localhost:8000/health | jq '.status'

# 5. Run smoke test
distllm benchmark run --num-prompts 3 --max-tokens 10
```

### Rolling Back a Configuration Change

**Scenario**: Config change causes performance degradation.

**Steps**:
```bash
# 1. Restore from backup
distllm backup list
distllm backup restore <backup-id>

# 2. Or manually revert config.yaml
git checkout HEAD~1 -- config.yaml

# 3. Hot-reload config (no restart needed)
kill -HUP $(pgrep -f distllm-api)

# 4. Verify
curl http://localhost:8000/v1/scheduler/config
```

### Rolling Back a Kubernetes Deployment

**Scenario**: Bad deployment in Kubernetes.

**Steps**:
```bash
# 1. Check rollout history
kubectl rollout history deployment/distllm-coordinator -n distllm

# 2. Rollback to previous revision
kubectl rollout undo deployment/distllm-coordinator -n distllm

# 3. Rollback to specific revision
kubectl rollout undo deployment/distllm-coordinator -n distllm --to-revision=3

# 4. Verify pods are healthy
kubectl get pods -n distllm -l app=distllm

# 5. Check health
kubectl exec -n distllm deploy/distllm-coordinator -- curl -s localhost:8000/health
```

### Rolling Back a Docker Compose Deployment

**Steps**:
```bash
# 1. Stop current deployment
docker compose down

# 2. Checkout previous version
git checkout v0.3.0

# 3. Rebuild and restart
docker compose up -d --build

# 4. Verify
docker compose logs -f coordinator
```

---

## Scaling Operations

### Adding a Worker Node

```bash
# 1. Start new worker
distllm-node --coordinator coordinator:50050 --port 50053 \
  --model meta-llama/Llama-2-70b --start-layer 48 --end-layer 63 --total-layers 80

# 2. Verify registration
curl http://localhost:8000/admin/v1/nodes | jq '.nodes | length'

# 3. Rebalance layers (if needed)
curl -X POST http://localhost:8000/admin/v1/nodes/rebalance
```

### Removing a Worker Node (Graceful Drain)

```bash
# 1. Drain the node (stop sending new requests)
curl -X POST http://localhost:8000/admin/v1/nodes/{node_id}/drain

# 2. Wait for in-flight requests to complete
sleep 30

# 3. Verify no active requests
curl http://localhost:8000/admin/v1/nodes/{node_id} | jq '.active_requests'

# 4. Remove from cluster
curl -X POST http://localhost:8000/admin/v1/nodes/{node_id}/offline

# 5. Stop the worker process
kill $(pgrep -f "distllm-node.*{node_id}")
```

---

## Incident Response

### High Latency (P99 > 5s)

1. Check GPU utilization: `nvidia-smi dmon -s u`
2. Check queue depth: `curl http://localhost:8000/v1/scheduler/stats`
3. Check for stragglers: `distllm cluster status`
4. If GPU > 90%: Scale up workers
5. If queue depth > 100: Increase `max_batch_size` or add nodes
6. If single node slow: Check for thermal throttling (`nvidia-smi -q -d TEMPERATURE`)

### Memory OOM

1. Check KV cache usage: `curl http://localhost:8000/metrics | grep kv_cache`
2. Enable defragmentation: `distllm defrag run`
3. Reduce KV cache: `export DISTLLM_KV_CACHE_QUANT_BITS=8`
4. Reduce batch size: `curl -X PATCH http://localhost:8000/v1/scheduler/config -d '{"max_batch_size": 8}'`
5. If persistent: Add nodes or use smaller model

### Node Unreachable

1. Check node health: `curl http://localhost:8000/admin/v1/nodes/{id}`
2. Check network: `ping {node_host}` and `nc -zv {node_host} {port}`
3. Check node logs: `journalctl -u distllm-node -f`
4. If gRPC timeout: Check TLS configuration
5. If node crashed: `systemctl restart distllm-node`
6. If persistent: Drain and remove node, add replacement

---

## Maintenance Windows

### Upgrading DistLLM Version

```bash
# 1. Create backup
distllm backup create --type full

# 2. Drain all nodes
for node in $(curl -s http://localhost:8000/admin/v1/nodes | jq -r '.nodes[].node_id'); do
  curl -X POST http://localhost:8000/admin/v1/nodes/$node/drain
done

# 3. Wait for in-flight requests
sleep 60

# 4. Stop services
docker compose down  # or systemctl stop distllm-*

# 5. Upgrade
pip install --upgrade distributed-llm

# 6. Run migrations (if any)
distllm config validate

# 7. Start services
docker compose up -d  # or systemctl start distllm-*

# 8. Verify
distllm doctor
curl http://localhost:8000/health
```

### Certificate Rotation

```bash
# 1. Generate new certificates
distllm security cert generate --cn coordinator.example.com

# 2. Distribute to all nodes
for node in node1 node2 node3; do
  scp certs/coordinator.* $node:/etc/distllm/tls/
done

# 3. Reload (no restart needed)
kill -HUP $(pgrep -f distllm-api)
kill -HUP $(pgrep -f distllm-node)

# 4. Verify TLS
openssl s_client -connect coordinator:50050 -servername coordinator
```

---

## Monitoring Checklist

### Daily
- [ ] Check `/health` endpoint
- [ ] Review error rates in Grafana
- [ ] Check GPU utilization trends
- [ ] Verify backup completion

### Weekly
- [ ] Review SLO compliance (P95 latency, error rate)
- [ ] Check disk usage for logs and backups
- [ ] Review security alerts
- [ ] Test restore from backup

### Monthly
- [ ] Run full benchmark suite
- [ ] Review and rotate API keys
- [ ] Update dependencies
- [ ] Review capacity planning projections

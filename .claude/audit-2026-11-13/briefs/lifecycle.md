# Fix brief: lifecycle

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [high] src/distllm/api/routes/admin.py:322
**Summary**: `offline` and `recover` branch on attrs that real nodes do not have (`healthy`/dict), so offline is a silent no-op and recover cannot clear `is_healthy`; both return 200.
**Failure scenario**: `coord.nodes` holds PipelineNode objects (dataclass field is `is_healthy`, no `healthy`). offline_node: the `isinstance(node, dict)` branch does nothing, the `hasattr(node, 'healthy')` branch is never taken for PipelineNode, so it only logs "cannot mark node ... offline (unknown type: PipelineNode)" — verified: after a 200 offline the node was still routed to by run_pipeline and `node.is_healthy` stayed True; the circuit-breaker simulation it performs is never consulted by run_pipeline and is erased by record_success on the next successful request. recover_node for a PipelineNode executes `node.healthy = True` — a dead attribute; verified: with `node.is_healthy=False` (as set by a failed register/health path), recover 200s but is_healthy remains False and routing still raises 'No healthy nodes in pipeline'. So there is no admin verb that exits the "pipeline-unhealthy" state.
**Check command**: `cd "D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper" && python .claude/tmp_node_lifecycle_audit.py  (prints: [OK ] BUG: request still routed to offlined node; [OK ] BUG: recover does not restore is_healthy; offline warning 'unknown type: PipelineNode')`
**Reproduced by auditor**: True

## [med] src/distllm/api/routes/admin.py:390
**Summary**: Deregister leaves the id in `_draining_nodes` forever (no removal path), so a rejoined node with the same id reports as draining and the drain metric stays inflated.
**Failure scenario**: DELETE /admin/v1/nodes/{id} calls mark_node_draining then pops the node; the only code that discards an id from `_draining_nodes` is the circuit-breaker cooldown (needs failure counts >= 3 plus a recovery time — admin drain creates neither) and `mark_node_alive`, reachable via undrain which 404s once the node is gone. Verified: after deregister, `is_node_draining('w1')` is True, `get_metrics()['draining_nodes']` is 1 (feeds the distllm_draining_nodes gauge), `POST .../undrain` returns 404, and after re-registering the same node id `GET /admin/v1/nodes` shows `draining: true, state: 'draining'` and cluster list-nodes prints '1 draining' — the exact a10-worker-1 symptom the delete route was added to fix, now re-created through delete itself.
**Check command**: `cd "D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper" && python .claude/tmp_node_lifecycle_audit.py  (prints: draining set after delete: ['w1']; metrics draining_nodes: 1; undrain 404; 'BUG: rejoined node shows draining=True')`
**Reproduced by auditor**: True

## [med] src/distllm/api/routes/admin.py:159
**Summary**: Admin node listing reads `getattr(node, 'healthy', False)` on PipelineNode (which has `is_healthy`), so every node lists healthy=false, healthy_count is always 0, and cluster/status is permanently 'degraded'.
**Failure scenario**: Verified with a real registered PipelineNode: `GET /admin/v1/nodes` returned `healthy: false` (healthy_count=0) while the same payload's `state` said 'healthy' and the node was serving, and `GET /admin/v1/cluster/status` returned status 'degraded', healthy_nodes=0. Any operator or dashboard consuming the admin API sees a constantly degraded cluster; the self-contradictory healthy/state fields also mask genuinely stuck nodes (and are how finding #1's stuck node appears: healthy=false + state=healthy).
**Check command**: `cd "D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper" && python .claude/tmp_node_lifecycle_audit.py  (prints: listing healthy=False state=healthy; [OK ] healthy_count=0 despite live node; [OK ] cluster status degraded despite live node)`
**Reproduced by auditor**: True

## [med] src/distllm/cli/main.py:977
**Summary**: `distllm cluster leave` defaults `--port 50050` (the gRPC port) but `_cluster_leave` calls REST admin routes (which live on 8000), so the command cannot work against a default cluster.
**Failure scenario**: `cluster list-nodes` defaults to --port 8000 with an explicit note it talks to the REST port; `cluster leave` (and `cluster drain`) create `_get_client(host, 50050)` for /admin/v1/nodes/... Against a default `cluster start`, port 50050 speaks gRPC, not HTTP; against an idle port it prints 'Could not connect ... 50050'. Reproduced against a plain TCP listener: the run ended with raw 'Error: [WinError 10053] An established connection was aborted...' via the generic CLI error handler instead of the friendly admin-path message, and both the drain/delete 401 handling and 'node is not registered' text can never trigger because the request never reaches the admin router. Fix works only by manual `--port 8000`.
**Check command**: `cd "D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper" && python -m distllm.cli.main cluster leave --help  (shows --port default 50050 for the REST admin path; contrast: python -m distllm.cli.main cluster list-nodes --help shows 8000)`
**Reproduced by auditor**: True

## [med] src/distllm/core/coordinator.py:1699
**Summary**: Coordinator restart loses all lifecycle state: `_draining_nodes` is in-memory and never persisted, shutdown_state.json is write-only, and HA snapshot application discards node health.
**Failure scenario**: A node drained just before a coordinator restart comes back fully listed and routable: ResourceManager state lives only in memory (verified: fresh ResourceManager reports draining=False after the previous one had True), `ResourceManager.save_state/load_state` exist but are never called anywhere in src/, `_save_shutdown_state` writes ~/.distllm/shutdown_state.json that no code ever reads (the only other reference is its test), and workers register only once at process start (no heartbeat/re-register in dist/worker.py; the failover handler is unwired), so after a coordinator restart the node pool is rebuilt only if workers are restarted too. On HA failover the snapshot carries per-node 'healthy' but apply_state_snapshot (coordinator_election.py:161-171) re-registers nodes without re-applying health, so previously unhealthy or draining nodes silently return to service.
**Check command**: `cd "D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper" && python .claude/tmp_drain_restart.py  (prints: pre-restart draining: True; post-restart draining (fresh RM): False)  ; grep -rn "save_state\(|load_state\(|shutdown_state" src/ | grep -v model_version`
**Reproduced by auditor**: True

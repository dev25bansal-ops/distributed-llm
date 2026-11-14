# Fix brief: keys

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [crit] src/distllm/dist/worker.py:569
**Summary**: worker main() crashes with AttributeError when any config.yaml exists: settings.network.cluster_key does not exist (NetworkSettings has no cluster_key field), so `distllm cluster join` workers die at startup; the join CLI then waits 60s and misreports 'Node registered with coordinator' while the worker is dead.
**Failure scenario**: Run `distllm cluster join` from any directory containing config.yaml (the repo root ships one; `distllm config setup` writes one). The spawned worker resolves ConfigResolver._resolve_config_path('worker') -> config.yaml, loads DistLLMSettings, reaches line 568-569 `cluster_key = settings.network.cluster_key or None` and raises AttributeError: 'NetworkSettings' object has no attribute 'cluster_key'. The worker exits before registering; the CLI's port-poll times out ('registering anyway'), its own admin registration still returns 200, so the node appears in `cluster list-nodes` but no worker serves it. Verified: running the worker from the worktree root reproduced the exact AttributeError (worktree line 569).
**Check command**: `cd <repo-root> && PYTHONPATH=src python -m distllm.dist.worker --node-id p --model m --start-layer 0 --end-layer 1 --total-layers 2 --port 59999   # AttributeError: 'NetworkSettings' object has no attribute 'cluster_key'`
**Reproduced by auditor**: True

## [high] src/distllm/cli/chat_stream.py:224
**Summary**: `distllm chat --stream` builds a bare httpx.AsyncClient with no Authorization header and no call to resolve_api_key/auth_headers anywhere in the module, so streaming chat 401s against every server (auth is always required), while the non-streaming `distllm chat` path works because it uses client_for().
**Failure scenario**: Start a cluster (key only in .distllm/api_key), run `distllm chat --stream`, send a message: _stream_chat POSTs /v1/chat/completions without any Authorization header and the server returns 401; the CLI prints 'API error (401)'. Reproduced against a stub server on port 8934: chat_stream sent Authorization=None and got 401, while client_for() sent 'Bearer sk-from-file'. `grep -c -E 'Authorization|auth_headers|resolve_api_key' src/distllm/cli/chat_stream.py` returns 0, confirming no auth code path exists in the module.
**Check command**: `grep -c -E "Authorization|auth_headers|resolve_api_key" src/distllm/cli/chat_stream.py  # 0 matches: the streaming path never authenticates`
**Reproduced by auditor**: True

## [high] src/distllm/core/api_key_store.py:529
**Summary**: The server-side key store and auth middleware never read DISTLLM_API_KEY (only API_KEYS/API_KEYS_FILE/API_KEY), while docs tell operators to export DISTLLM_API_KEY and every CLI/SDK client prefers DISTLLM_API_KEY over the working key file - so the documented env var can never configure the server and silently shadows the key file that would authenticate.
**Failure scenario**: Follow docs/SECURITY_HARDENING.md:186 (`export DISTLLM_API_KEY=sk-documented`) and start the server. ApiKeyStore._load (line 529 reads only API_KEY) finds nothing, auto-generates a different key into .distllm/api_key, and authenticate('sk-documented') returns None. Meanwhile cli/http_client.resolve_api_key (line 30) prefers DISTLLM_API_KEY, so every CLI command sends sk-documented and 401s with 'Unauthorized: invalid API key' even though the correct key is on disk. With both API_KEY=A and DISTLLM_API_KEY=B set, the store authenticates A only while clients send B (verified: store.authenticate('sk-A')=True, 'sk-B'=None, resolve_api_key()='sk-B').
**Check command**: `PYTHONPATH=src DISTLLM_API_KEY=sk-x python -c "import os,tempfile; os.environ['DISTLLM_KEY_FILE']=tempfile.mkdtemp()+os.sep+'k'; from distllm.core.api_key_store import ApiKeyStore; s=ApiKeyStore(); print('authenticates DISTLLM_API_KEY value:', s.authenticate('sk-x'))"  # prints None; then: PYTHONPATH=src DISTLLM_API_KEY=sk-x python -c "from distllm.cli.http_client import resolve_api_key; print(resolve_api_key())"  # prints sk-x`
**Reproduced by auditor**: True

## [med] src/distllm/cli/client.py:71
**Summary**: DistLLMClient (used by `distllm benchmark eval` and federated commands) resolves the key only from DISTLLM_API_KEY - it never reads API_KEY nor the coordinator key file - so the default eval flow 401s against a cluster started by `distllm cluster start` (whose admin key lives only in the key file).
**Failure scenario**: Start a cluster with `distllm cluster start` (server mints the key into .distllm/api_key; no env vars). Run `distllm benchmark eval my-model` without --api-key: cli/eval.py:77 sets api_key_resolved='' and constructs DistLLMClient(api_key=''), whose constructor falls back only to os.environ['DISTLLM_API_KEY'] -> no Authorization header -> POST /v1/eval/run returns 401. Verified: with API_KEY='sk-file' and the key file holding the same value, DistLLMClient._headers contained no Authorization.
**Check command**: `PYTHONPATH=src python -c "import os,tempfile; os.environ['DISTLLM_KEY_FILE']=tempfile.mkdtemp()+os.sep+'k'; open(os.environ['DISTLLM_KEY_FILE'],'w').write('sk-file'); os.environ['API_KEY']='sk-file'; from distllm.cli.client import DistLLMClient; print(DistLLMClient(base_url='http://x')._headers)"  # no Authorization key`
**Reproduced by auditor**: True

## [med] src/distllm/cli/cluster.py:399
**Summary**: Same CLI, two different env orders: cluster join registration reads API_KEY first then DISTLLM_API_KEY, while http_client.resolve_api_key and cluster._admin_headers read DISTLLM_API_KEY first (and dist/p2p/transport.py reads only DISTLLM_GOSSIP_API_KEY then API_KEY), so when both vars are set to different values different commands send different keys.
**Failure scenario**: Set API_KEY=A and DISTLLM_API_KEY=B (common during the v0.3->v0.4 migration the docs describe). The server (API_KEY-only, per finding 2) accepts A. `distllm cluster join` sends A (works), while `distllm cluster list-nodes`, `distllm chat`, and `distllm model list` send B -> 401. Verified: resolve_api_key()='sk-B', _admin_headers()={'Authorization': 'Bearer sk-B'}, join path picks 'sk-A'. dist/worker.py lines 453/488 also pick API_KEY first (opposite of the clients that talk to the same server).
**Check command**: `API_KEY=A DISTLLM_API_KEY=B PYTHONPATH=src python -c "import os; from distllm.cli.http_client import resolve_api_key; from distllm.cli.cluster import _admin_headers; print('resolve:', resolve_api_key(), '| admin:', _admin_headers().get('Authorization'), '| join picks:', os.environ.get('API_KEY') or os.environ.get('DISTLLM_API_KEY'))"`
**Reproduced by auditor**: True

## [med] src/distllm/dist/worker.py:453
**Summary**: Worker's own coordinator registration and ready-notification never read the coordinator key file (the same bug class as the known-fixed cli/cluster.py join): they send `Bearer <cluster_key>` when a cluster key is configured, or only env API_KEY/DISTLLM_API_KEY, so with the default file-only key the worker's registration/ready calls 401.
**Failure scenario**: Start a cluster with `distllm cluster start` (key only in .distllm/api_key) and run `distllm cluster join` with no env keys: dist/worker.py _register_with_coordinator(None) and _notify_coordinator_ready(None) send headers={'Content-Type': 'application/json'} - no Authorization - and the coordinator answers 401 (verified with a stub server: both calls captured no Authorization header and returned 401). The CLI-side registration masks this, but the worker's ready signal never lands, and a directly-started worker (python -m distllm.dist.worker on another host) never registers at all. With --cluster-key K, the worker sends Bearer K, which the API key store does not accept (the store only honors API_KEYS/API_KEYS_FILE/API_KEY), causing the same 401.
**Check command**: `grep -n "DISTLLM_KEY_FILE\|resolve_api_key\|_api_key_path" src/distllm/dist/worker.py   # no matches: worker has no key-file read; only os.environ.get('API_KEY') or os.environ.get('DISTLLM_API_KEY') at lines 453, 488`
**Reproduced by auditor**: True

## [low] src/distllm/cli/audit.py:191
**Summary**: cli-doctor checks the wrong key-file location: it looks for ~/.distllm/api_key while the server and clients read the cwd-relative .distllm/api_key (or DISTLLM_KEY_FILE), so it reports 'generated' when the server will actually load the file, and check_auth reads only API_KEY (never DISTLLM_API_KEY or the file).
**Failure scenario**: Start a server in a directory with no env vars; it writes .distllm/api_key. Run `distllm cli-doctor` from that directory: check_api_key_store checks os.path.expanduser('~/.distllm/api_key') (line 191), which does not exist, and reports 'API key source: generated' - contradicting the key file the server actually uses. Verified: with HOME redirected to an empty dir and a real .distllm/api_key in cwd, the doctor verdict was 'generated'.
**Check command**: `PYTHONPATH=src python -c "from distllm.core.api_key_store import _api_key_path; print('server reads:', _api_key_path())" && grep -n "distllm/api_key" src/distllm/cli/audit.py src/distllm/core/api_key_store.py  # audit.py uses ~/.distllm/api_key, store uses .distllm/api_key`
**Reproduced by auditor**: True

## [low] src/distllm/core/framework_adapters.py:52
**Summary**: Framework adapters and integration packages resolve only DISTLLM_API_KEY (or a fabricated placeholder) and never fall back to API_KEY or the key file: with no DISTLLM_API_KEY, _resolve_api_key returns the literal 'not-needed' and every adapter request 401s against a secured server.
**Failure scenario**: Choose SECURITY/QUICKSTART flow: API_KEY (or the auto key file) is the working credential. Call get_openai_client() / get_langchain_llm() with no explicit api_key: _resolve_api_key(None) returns 'not-needed' (verified), so the adapter sends 'Authorization: Bearer not-needed' and the server returns 401. The same DISTLLM_API_KEY-only pattern is in integrations/* (agno, autogpt, openai-agents, dify, ollama_compat) and in sdk/src/distllm_sdk/cli.py:97 and mcp_server.py:61.
**Check command**: `PYTHONPATH=src python -c "from distllm.core.framework_adapters import _resolve_api_key; print(repr(_resolve_api_key(None)))"  # 'not-needed' when DISTLLM_API_KEY is unset`
**Reproduced by auditor**: True

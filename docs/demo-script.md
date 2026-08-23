# DistLLM — 3-Minute Investor Demo Script

Recording script for the marketing/investor video: one coordinator machine +
two worker devices joining over LAN, a streaming chat completion, and the live
dashboard. Every step lists the exact command and the output you should see —
if a step doesn't match, pause and fix before recording the next take.

**Model:** `roneneldan/TinyStories-1M` (8 transformer layers) — small enough
that CPU-only laptops join in seconds, which keeps the recording snappy.

---

## Pre-flight checklist (do once, before rolling)

| # | Check | Command / action |
|---|-------|------------------|
| 1 | DistLLM installed on all three machines (`distllm --version` works) | `pip install -e .` from a repo checkout |
| 2 | Same repo version on coordinator + both workers | `git rev-parse --short HEAD` on each |
| 3 | All three machines on the same Wi-Fi/LAN; coordinator IP known | `ipconfig` (Windows) / `ifconfig` (macOS/Linux) |
| 4 | Ports free on coordinator: **50050** (gRPC), **8000** (REST) | close anything bound to them |
| 5 | Shared auth env vars exported in every terminal you'll show | see Step 0 |

Environment used throughout this script (set on ALL machines):

```bash
export DISTLLM_CLUSTER_KEY="demo-cluster-key"   # shared worker<->coordinator auth
export API_KEY="demo-admin-key-123"             # REST auth; this key gets the admin role
```

Set these BEFORE starting anything — the coordinator reads them at startup.

> Recording tip: use a clean terminal profile, font size 16+, and
> `export PS1="demo $ "` so the shell looks uncluttered on camera.

---

## Step 0 · Cold open (0:00 – 0:20)

**Say:** "This is DistLLM. What you're about to see: three ordinary laptops
team up to run one language model as a single pipeline — no cloud, no GPUs."

Nothing to run yet — just the talking head or title card.

---

## Step 1 · Start the coordinator (0:20 – 0:55)

On **machine A** (the coordinator):

```bash
distllm cluster start --model roneneldan/TinyStories-1M --api-port 8000
```

**Expected output:**

```
Starting coordinator...
  Model: roneneldan/TinyStories-1M
  gRPC port: 50050
  API port: 8000
  Mode: distributed
Coordinator started (PID: 12345)
Coordinator ready on port 50050
API server started (PID: 12346)
API server ready on port 8000
```

The command stays in the foreground running both processes — leave that
terminal visible but switch to a second terminal for the next commands.

**Verify before moving on** (second terminal, still on machine A):

```bash
curl -s http://localhost:8000/v1/health -H "Authorization: Bearer $API_KEY"
```

**Expected:** HTTP 200 and a JSON body like:

```json
{"status": "ok", "model": "roneneldan/TinyStories-1M", "nodes": 0}
```

`"nodes": 0` is the beat to call out on camera: *"Empty cluster. Watch what
happens when devices join."*

If the health call returns 401, the server didn't pick up `$API_KEY` —
restart the coordinator with the env var exported.

---

## Step 2 · Two worker devices join (0:55 – 1:35)

On **machine B**:

```bash
distllm cluster join --coordinator <COORDINATOR_IP> --port 50050 \
    --node-id laptop-1 \
    --model roneneldan/TinyStories-1M \
    --start-layer 0 --end-layer 3 --total-layers 8 \
    --device cpu
```

**Expected output:**

```
Joining cluster at 192.168.1.10:50050...
  Node ID: laptop-1  Listen port: 50051
Worker started (PID: 23456)
Connected to 192.168.1.10:50050
Waiting for worker to finish loading model...
Registered with coordinator
```

(The final registration line appears after model load completes — a few
seconds on CPU with TinyStories.)

On **machine C**, same command with `--node-id laptop-2 --start-layer 4
--end-layer 7`.

**Say while they join:** "Laptop one takes layers zero through three. Laptop
two takes four through seven. Together they are now one eight-layer model."

**Verify on the coordinator:**

```bash
distllm cluster list-nodes --coordinator localhost --port 8000
```

**Expected:** a rich table listing `laptop-1` and `laptop-2`, both healthy,
with their layer ranges and GPU/CPU info:

```
┌──────────┬─────────┬────────────┬───────────┬─────────┐
│ Node ID  │ Healthy │ Layers     │ Device    │ Memory  │
│ laptop-1 │ yes     │ 0 – 3      │ cpu       │ …       │
│ laptop-2 │ yes     │ 4 – 7      │ cpu       │ …       │
└──────────┴─────────┴────────────┴───────────┴─────────┘
```

Fallback if `list-nodes` isn't cooperating on camera: `curl -s
http://localhost:8000/admin/v1/nodes -H "Authorization: Bearer $API_KEY"`
shows the same data as JSON (see Step 3).

---

## Step 3 · The control plane: admin nodes + dashboard (1:35 – 2:05)

Still on the coordinator:

```bash
curl -s http://localhost:8000/admin/v1/nodes -H "Authorization: Bearer $API_KEY"
```

**Expected (abridged):**

```json
{
  "nodes": [
    {"node_id": "laptop-1", "healthy": true, "draining": false,
     "host": "192.168.1.11", "start_layer": 0, "end_layer": 3},
    {"node_id": "laptop-2", "healthy": true, "draining": false,
     "host": "192.168.1.12", "start_layer": 4, "end_layer": 7}
  ],
  "total_nodes": 2,
  "healthy_count": 2,
  "total_layers": 8
}
```

Then open the browser full-screen:

```
http://localhost:8000/dashboard
```

**Expected:** the real-time monitoring dashboard renders — throughput chart,
latency graph, system health tiles — connected over WebSocket. Hover a tile
or let a request land on it during Step 4 so the charts visibly move.

**Say:** "Every node reports heartbeats; the dashboard and the admin API read
live state. This is the ops view your team gets on day one."

Note: `/admin/v1/*` requires an admin-role key — that's why we exported
`API_KEY` before starting the server. A 403 here means the request went out
without `Authorization: Bearer $API_KEY`; a 401 means the key wasn't set at
server start.

---

## Step 4 · Streaming chat completion across the pipeline (2:05 – 2:50)

The money shot — a token stream flowing through both laptops:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "distributed-llm",
        "messages": [{"role": "user", "content": "Once upon a time, a little robot"}],
        "max_tokens": 60,
        "temperature": 0.8,
        "stream": true
      }'
```

**Expected:** Server-Sent Events arriving incrementally (with `-N` curl
prints each chunk as it lands):

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"there"},"index":0}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"was"},"index":0}]}

...

data: [DONE]
```

Tokens should appear over several seconds — that's activations physically
hoping laptop-1 → laptop-2 per generated token. Let 10+ chunks print before
moving on.

Optional second beat (non-streaming, shows the OpenAI-compatible shape):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "distributed-llm",
       "messages": [{"role": "user", "content": "Write a story about a dragon."}],
       "max_tokens": 30}' | python -m json.tool
```

**Expected:** standard chat completion object —

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}]
}
```

**Say:** "One request in, OpenAI-compatible streaming out — split across two
consumer laptops. No code change for the client."

---

## Step 5 · Close (2:50 – 3:00)

**Say:** "That's DistLLM: pip install, three laptops, one model. Private by
design — nothing leaves your network. We're opening the beta soon."

End card: repo URL + docs link.

---

## Troubleshooting during recording

| Symptom | Likely cause | Fix |
|---|---|---|
| Worker prints `No coordinators found on LAN` | joined with `--discover` but mDNS blocked | pass `--coordinator <IP>` explicitly |
| Worker stuck at `Waiting for worker to finish loading model...` | first run is downloading weights | run once beforehand so HF cache is warm |
| Health 401 / admin 403 | `API_KEY` not exported when server started | restart coordinator with env var set |
| `Connection refused` on 50050 | coordinator didn't start (port busy) | free the port, rerun Step 1 |
| Streaming returns non-SSE JSON | `"stream": true` typo'd | recheck the curl body |
| Dashboard blank | opened before API was ready | reload after Step 1 verification passes |

## Dry-run tip

Rehearse the whole flow once with the hermetic test suite instead of real
machines — it exercises the same HTTP surface in-process:

```bash
python -m pytest tests/dist/integration/test_wan_scenario.py -v
```

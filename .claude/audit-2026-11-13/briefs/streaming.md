# Fix brief: streaming

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [high] src/distllm/api/streaming.py:437
**Summary**: Streaming completions return a silent empty 200 (role chunk, empty finish_reason=stop, [DONE]) when the coordinator exists but has no partitioner and no pipeline nodes, instead of an error — the same 'silent empty 200' class as the previously fixed streaming import bug, via a different path.
**Failure scenario**: Run `distllm cluster start --model <m>` in the shipped distributed mode: the API server immediately has a Coordinator, but not a single worker has registered yet (or all workers have left). A client sends POST /v1/chat/completions with stream=true. `_generate_tokens` finds local_partitioner=None and node_order=[] and falls off the end without raising; `_stream_response` (line 437) only guards `not local_coord` and emits a role delta, an empty delta with finish_reason 'stop', and [DONE], HTTP 200. The caller (distllm chat --stream, any OpenAI SDK with stream=True) records a successful empty completion and has no way to know inference never ran. Reproduced over a live socket on port 8935: streaming returned HTTP 200 with only the role + stop + [DONE] chunks, while the identical non-streaming request on the same server correctly returned HTTP 500. Contrast: the non-stream path raises ConfigError('No nodes registered in the pipeline') from the engine, so only streaming silently 'succeeds'.
**Check command**: `python .claude/tmp-errmq/check_real_stream.py   (TestClient + real Coordinator in distributed mode; prints streaming 200 empty body vs non-streaming 500; no server needed)`
**Reproduced by auditor**: True

## [high] src/distllm/api/routes/health.py:618
**Summary**: POST /v1/update-params misreports and corrupts: it 404s for live streams (ParamUpdateChannel.update() always returns None) and, before raising, writes explicit None into unset fields of the live params, so the next token step raises and the stream truncates.
**Failure scenario**: A streaming client (the request id is visible in every chunk's 'id' field) calls the documented mid-stream steering endpoint with a partial body {temperature: 0.3}. Step 1 (reproduced): param_update_channel.py:71 'update()' has return type None and returns None, so the route raises 404 'Request ... not found or already completed' even though the request is live and the update landed. Step 2 (reproduced): the route passed top_p=None / top_k=None (unset fields), which params.update() applies via setattr, so the live params now hold None. Step 3 (reproduced): the next sampling step in the stream raises TypeError: '>' not supported between instances of 'NoneType' and 'int' (token_generator.py:43 top_k check / line 47 top_p check), and per the finding above the stream then truncates with no error signal — the client's own steering call killed its stream while telling it 404. A full three-field payload silently changes sampling while still reporting 404.
**Check command**: `cd /d/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python .tmp-sse-audit3.py  # R2a: HTTPException=404 for a LIVE id; R2b: top_p=None top_k=None after route call; R2c: next sample raises TypeError`
**Reproduced by auditor**: True

## [med] src/distllm/api/routes/model_registry.py:221
**Summary**: POST /api/models/load and /api/models/unload return HTTP 200 with a bare {"error": ...} body when no coordinator is loaded or no load/unload method exists; the repo's own CLI (distllm model load/unload) only checks the HTTP status and then tells the user the operation succeeded.
**Failure scenario**: Host the ASGI app without its main() entrypoint (e.g. `uvicorn distllm.api.server:app`, or any embedding of the app), so g.coordinator is None. POST /api/models/load returns HTTP 200 with body {"error": "coordinator not available"} — not the standard {"error": {message,...}} envelope, and not a 4xx/5xx. `distllm model load llama-x` against this server (distllm/cli/models.py:182) passes raise_for_status() and prints 'Model loaded: llama-x'; same for `distllm model unload` (cli/models.py:196) and `distllm deploy` which prints 'Deployment request accepted.'. The user is told a model was loaded/deployed when nothing happened. Reproduced live: server on port 8935 with no coordinator answered HTTP 200 {"error":"coordinator not available"}, and the real CLI function printed 'Model loaded: llama-x'. Rows 253 (unload) and 228/260 (method-missing variants) share the defect.
**Check command**: `python .claude/tmp-errmq/serve.py  (background, port 8935) ; then `DISTLLM_API_KEY=k-test python .claude/tmp-errmq/check_cli.py` -> prints 'Model loaded: llama-x' / 'Model unloaded: llama-x' despite HTTP 200 {"error":"coordinator not available"}`
**Reproduced by auditor**: True

## [low] src/distllm/api/routes/model_registry.py:197
**Summary**: GET /api/models/registry/{model_id} returns HTTP 200 {"error": "model not found"} for a missing model instead of a 404, so any client that checks response.ok treats the failed lookup as a successful detail fetch.
**Failure scenario**: GET /api/models/registry/not-a-real-model on a live server returns HTTP 200 with body {"error":"model not found","id":"not-a-real-model"} (the 404 branch does not exist; the route's OpenAPI docs promise 'detailed info for a single model'). An external client or monitor that keys on the HTTP status (curl --fail, fetch response.ok, OpenAI-style SDK error handling) reads this as success and receives an error object where a model record was expected.
**Check command**: `python .claude/tmp-errmq/check_registry.py   (prints '200 application/json {"error":"model not found","id":"not-a-real-model"}')`
**Reproduced by auditor**: True

## [low] src/distllm/api/routes/webrtc.py:190
**Summary**: POST /v1/webrtc/ice returns HTTP 200 {"status":"error", ...} whenever a candidate cannot be applied (malformed candidate, missing transport, aiortc unavailable), so a signaling client checking the HTTP status believes the candidate was accepted; the route's documented responses (404, 422) promise no such 200-error shape.
**Failure scenario**: With a session created via POST /v1/webrtc/offer, POST /v1/webrtc/ice with candidate='x', sdp_mid='' returns HTTP 200 with body {"status":"error","detail":"Candidate must have either sdpMid or sdpMLineIndex"}. Reproduced live on port 8935. A browser signaling client (or any HTTP client that keys on status code) sees 200 and proceeds assuming the candidate was registered, while ICE connectivity will silently fail; the route's OpenAPI entry documents only 404/422 for this endpoint. The offer handler has the sibling issue at line 152-155: any setup failure collapses to 500 detail='WebRTC setup failed', which never says whether the SDP, aiortc, or connectivity was at fault.
**Check command**: `python .claude/tmp-errmq/serve.py  (background, port 8935) ; curl -s -w ' HTTP:%{http_code}' -H 'Authorization: Bearer k-test' -H 'Content-Type: application/json' -d '{"sdp":"x","type":"offer","session_id":"s1"}' http://127.0.0.1:8935/v1/webrtc/offer ; curl -s -w ' HTTP:%{http_code}' -H 'Authorization: Bearer k-test' -H 'Content-Type: application/json' -d '{"session_id":"s1","candidate":"x","sdp_mid":"","sdp_mline_index":0}' http://127.0.0.1:8935/v1/webrtc/ice   -> HTTP:200 with status=error body`
**Reproduced by auditor**: True

## [low] src/distllm/api/server_errors.py:72
**Summary**: The catch-all 500 handler replaces every unhandled exception's message with 'An unexpected error occurred. Please try again later.', so a configuration error like 'No nodes registered in the pipeline' is reported to the caller as a transient failure with reusable retry advice.
**Failure scenario**: POST /v1/chat/completions (non-streaming) against a distributed coordinator with no registered nodes: the engine raises ConfigError('No nodes registered in the pipeline'), which reaches general_exception_handler and becomes HTTP 500 {"error":{"message":"An unexpected error occurred. Please try again later.","code":"config_error"}}. The message tells the caller to retry a configuration problem that will never succeed until nodes are added; the actionable cause exists only in the server log line 'Unhandled exception: [CONFIG_ERROR] No nodes registered in the pipeline'. Reproduced via both TestClient and a live socket (port 8935).
**Check command**: `python .claude/tmp-errmq/check_real_stream.py   (non-streaming section prints the 500 with the generic message; server-side log line shows the masked ConfigError)`
**Reproduced by auditor**: True

## [low] src/distllm/api/routes/chat.py:739
**Summary**: stream=true with max_tokens=0 returns a JSON ChatCompletionResponse instead of SSE (same shape in completion.py:118) — the stream flag is silently switched off.
**Failure scenario**: A client posts {stream:true, max_tokens:0} (natural when a caller's budget is clamped to 0 or during probes): the route returns before ever checking body.stream (reproduced route-level: returned ChatCompletionResponse with content-type application/json, not text/event-stream). An SSE reader (e.g. distllm/client/client.py stream_generate, which only parses 'data: ' lines) sees an empty stream and no [DONE], indistinguishable from a dropped connection.
**Check command**: `cd /d/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python .tmp-sse-audit3.py  # R4: returned ChatCompletionResponse (stream flag ignored)`
**Reproduced by auditor**: True

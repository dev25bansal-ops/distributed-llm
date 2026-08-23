---
tags:
  - audit
  - exhaustive
date: 2026-08-11
---

# Exhaustive Audit 07 — Integrations

**← [[Exhaustive Audit 2026-08-11]]**

All findings in category `integration` (Medium/Low and non-verified severities).

**4 findings** — Medium: 4

---

### F-229 — [Medium] Strengths/weaknesses: OpenAI Agents + Semantic Kernel + fastapi router are the strongest; Dify + grpc_client are the weakest; haystack/autogen/litellm/one-api are thin config-only wrappers with version-drift risk

`integrations/autogen/src/distllm_autogen/config.py:73` · zone=`integrations` · category=`integration`

- **Summary:** Best adapters: OpenAI Agents (model.py) correctly delegates to the framework's own OpenAIModel pointed at DistLLM, Semantic Kernel (chat_completion.py) uses correct /v1 paths and a proper SSE parser, and fastapi_fimodel router.py posts to '/v1/chat/completions' correctly. Weakest: gRPC client (wrong service name/missing proto) and Dify (v1 doubling). Config-only wrappers drift from current framework versions: Autogen uses the legacy `config_list`/`cache_seed` form (removed/renamed in pyautogen 0.4+), LiteLLM sets the deprecated `litellm.custom_provider_map` and always hardcodes `stream=False` so streaming requests silently return the full response, and Haystack's OpenAIGenerator/embedder subclassing pins to older component class names.
- **Evidence (verbatim):**
```
"cache_seed": None,  # Disable caching for distributed inference
```
- **Impact:** Support burden: users on current framework versions (autogen 0.4+, litellm 1.5x, haystack 2.x) hit config keys/classes that are deprecated or moved, degrading the '20+ integrations' value proposition.
- **Effort:** 1-2 days
- **Recommendation:** Add explicit minimum-version pins and CI smoke tests in each package's pyproject that import and construct the adapter against a current framework release; move LiteLLM to litellm.custom_llm_provider registration and wire stream passthrough; update Autogen to the modern config shape.
- **Strategic value:** The breadth-of-integration claim is a primary differentiator; an integration matrix CI (construct+version) converts thin wrappers from liability into a verified compatibility story.

---

### F-230 — [Medium] pyproject: plugins/airflow.py and kubeflow.py need 'requests' but it is not a declared dependency

`pyproject.toml:37` · zone=`ops-utils` · category=`integration`

- **Summary:** DistLLMBatchOperator.execute and kubeflow_batch_inference_op call requests.post/get, but pyproject dependencies (lines 37-50) list fastapi/uvicorn/pydantic/httpx/numpy/etc. with no 'requests'. Since imports are lazy (inside execute), module import succeeds but the operator fails at runtime with ModuleNotFoundError on a default install; httpx and fastapi do not guarantee requests transitively.
- **Evidence (verbatim):**
```
dependencies = ["fastapi","uvicorn","pydantic",..."httpx>=0.27",...] (lines 37-50) contains no requests; airflow.py.py "import requests" is inside execute() (line 71)
```
- **Impact:** Airflow and Kubeflow batch operators are broken out-of-the-box for users on a standard pip install, surfacing only at DAG runtime.
- **Effort:** 1-2 hours
- **Reliability:** pip install distributed-llm (default extras) then call DistLLMBatchOperator.execute() -> ModuleNotFoundError: No module named 'requests'.
- **Recommendation:** Add 'requests>=2.x' to dependencies (simplest) or refactor airflow/kubeflow to use httpx which is already a dependency; add a lightweight import test that executes a mocked submission.

---

### F-231 — [Medium] TGI and Ollama adapters violate the BackendAdapter contract (async generate, dict return, load_model signature, non-classmethod display_name)

`src/distllm/backends/tgi_backend.py:96` · zone=`backends-config-cloud` · category=`integration`

- **Summary:** Base BackendAdapter defines sync generate(prompt,...) -> str and classmethod display_name(); TGI and Ollama override generate with async def ... -> dict, TGI overrides load_model(self, model_name) (base is load_model(self)), and Ollama knocks display_name into an instance @property. Callers following the protocol (adapter.generate()) get a coroutine/dict instead of str, and NodeService calling load_model() with no args on TGI raises TypeError. These two are also absent from _register_builtins, so they exist outside the registry path entirely.
- **Evidence (verbatim):**
```
async def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, ...) -> dict[str, Any]:
```
- **Impact:** Any shared pipeline that dispatches through the protocol breaks type/semantics contract for TGI/Ollama; also unregisterable from the normal flow.
- **Effort:** 3-4 hours
- **Reliability:** tgi_backend.py:96 and ollama_backend.py:28 are async and return dicts; ollama_backend.py:20-25 turns classmethods into instance property; _register_builtins list (backends/__init__.py:93-104) omits both.
- **Recommendation:** Align TGI/Ollama to the protocol: sync generate() returning str (or add an async_backend Protocol and register them), fix load_model(self) signature, and move display_name to a @classmethod. Add them to the registry or explicitly exclude them with a documented rationale.

---

### F-232 — [Medium] DraftModelFleet acceptance-rate scoring is never populated on the main generation path, so min_acceptance_rate>0 always fails routing and acceptance_weight never matters

`src\distllm\core\distributed_speculative.py:1399` · zone=`core-decoding` · category=`integration`

- **Summary:** DraftModelRouter hard-filters on `health.recent_acceptance_rate >= constraints.min_acceptance_rate` (draft_model_router.py line 314) and scores using `health.recent_acceptance_rate or spec.avg_acceptance_rate`. But the fleet health is only updated by `_query_all_drafts` -> `record_success(url, latency_s=0.0, tokens_generated=...)`, which never passes `acceptance_rate` (distributed_speculative.py line 1399-1402), and `generate`/`agenerate` do not call record_success with acceptance at all. So `recent_acceptance_rate` stays 0.0, the hard filter rejects every candidate whenever min_acceptance_rate>0 (always falling back), and selection is driven purely by latency/cost/load — the advertised 'accuracy/acceptance' routing dimension is effectively dead.
- **Evidence (verbatim):**
```
self._fleet.record_success(url, latency_s=0.0, tokens_generated=len(result.token_ids),)  # acceptance_rate omitted -> stays default
```
- **Impact:** min_acceptance_rate constraint is unsatisfiable (permanent fallback routing) and Acceptance routing weight is a no-op, so the SLA/accuracy routing guarantee the fleet advertises does not function in the distributed path.
- **Reliability:** Code trace: only two calls to record_success in src (distributed_speculative.py 1399); neither passes acceptance_rate; DraftModelHealth.recent_acceptance_rate default 0.0 used unchanged by DraftModelRouter line 314/375.
- **Recommendation:** Plumb the verified acceptance ratio into record_success from the caller that actually knows it (pass `acceptance_rate=accepted_count/len(draft_token_ids)` in generate/agenerate, and from _query_all_drafts after verification). Or compute recent_acceptance_rate inside DraftModelHealth from total_accepted when callers provide it; add a param and wire it at both call sites.

---

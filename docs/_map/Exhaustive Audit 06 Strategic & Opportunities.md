---
tags:
  - audit
  - exhaustive
date: 2026-08-11
---

# Exhaustive Audit 06 — Strategic & Opportunities

**← [[Exhaustive Audit 2026-08-11]]**

All findings in category `strategic, enhancement, new_feature` (Medium/Low and non-verified severities).

**14 findings** — High: 6 · Medium: 5 · Low: 3

---

### F-215 — [High] Top 5 sequenced strategic moves for differentiation (in order), each with market angle

`TASKS.md:69` · zone=`strategic` · category=`strategic`

- **Summary:** Based on the actual build state (TASKS.md progress, perf data, monetization docs vs code), the high-signal sequencing is: 1) CLOSE THE PERF GAP (single-node qualifier + unfilled TTFT/ITL) and repivot messaging to 'runs what you can't run / data-local / near-zero cost' rather than 'faster' — this is the precondition for every install. 2) OLLAMA-DROP-IN CLUSTERING (a 'distllm cluster' over the existing llama.cpp backend such that `ollama pull`/`ollama run` users get a real multi-machine mode) — taps the 175K-star single-device base whose docs/COMPETITIVE says is the 'next logical step' demographic. 3) ONE-COMMAND INSTALL + LIVE HF-SPACES DEMO (TASK-011/012, both unstarted) — convert docs-to-users and give YC/enterprise a proof point. 4) SHIP THE GPU MARKETPLACE (Hub) with escrow+reputation as the competitor-proof network moat — the only asset that cannot be easily cloned and that adds a real take-rate revenue line. 5) ENTERPRISE air-gap/SSO/audit tier — only after a paying pilot validates demand, so compliance build doesn't run ahead of revenue.
- **Evidence (verbatim):**
```
Strategic — Product & Traction (Weeks 8-12): TASK-011 One-command install ... TASK-012 HuggingFace Spaces demo ... TASK-013 TCO calculator ... TASK-015 Benchmark blog post
```
- **Impact:** Sequencing avoids the two classic startup failures on display here: polishing a huge feature surface (moonshot modules) while the baseline benchmark is empty, and writing enterprise/compliance docs before there is a paying customer.
- **Effort:** plan: weeks 1-3 = moves 1-3; move 4 = next quarter; move 5 = on-demand
- **Recommendation:** Treat moves 1-3 as a single 'make the first run win' release (weeks), move 4 as the strategic revenue bet (quarter), and hold 5 until a real customer request exists. Sequence protects scarce founder time: performance and onboarding unlock all adoption; the marketplace is the moat engine; enterprise is deferred until pull.
- **Strategic value:** Maritime value: each move compounds. Perf unlocks installs; Ollama adjacency gives distribution faster than category creation; marketplace converts the technical moat into a network effect and recurring take-rate revenue competitors cannot copy. Tradeoff: moving 4 early risks building a two-sided marketplace before supply (GPU owners) and demand (renters) exist — good reason to gate it behind real OSS adoption signal from moves 1-2.

---

### F-216 — [High] Docs over-claim vs code reality is a credibility/trust risk: marketing (billing pkg, live demo, 'real users') is ahead of shipped reality

`docs/MARKETING.md` · zone=`strategic` · category=`strategic`

- **Summary:** The docs are unusually ambitious and self-referential: MARKETING references a billing module that does not exist; YC_STARTUP_PACKAGE asserts 'working product, real users' while the one-command installer, live HF Spaces demo, TCO calculator and RBAC matrix docs are all unstarted (TASKS 011-015); PERFORMANCE_COMPARISON and BENCHMARKS carry empty latency cells; EXPORT_CONTROLS, GDPR, TERMS, CLOUD_MARKETPLACE, SLA_TIERS and a full compliance/ tree imply an enterprise- and marketplace-readiness the shipped code has not reached. For a startup whose OSS adoption is the top-of-funnel, a technical evaluator who discovers a claimed feature is absent loses trust in everything — including the security posture marketing relies on.
- **Evidence (verbatim):**
```
# Simplified model — actual implementation is in src/distllm/billing/ ... src/distllm/billing/ does not exist; TASK-011..015 (install, HF demo, TCO, RBAC docs, benchmark) all unstarted in TASKS.md
```
- **Impact:** Directly protects conversion (technical evaluators abort on any discovered over-claim) and investor/YC diligence, both of which read docs before code. It also exposes the real gap (monetization unbuilt) that the founder should sequence toward.
- **Effort:** 1 week (status matrix + EXPERIMENTAL marks + doc trim of unshipped claims)
- **Recommendation:** Add a concise 'Status / What's shipped vs planned' matrix near the top of README and the docs index (the vault already has an 11-Platform-Services _map — reuse it) so a contributor or evaluator can tell built from aspirational at a glance. Mark EXPERIMENTAL on the parked moonshot modules (TASK-018), and do NOT publish 'real users' or 'billing available' until true. This is cheap and directly protects the trust that the marketing funnel depends on.
- **Strategic value:** Trust is the scarcest asset in an OSS infra startup. Keeping docs exactly equal to shipped reality converts early adopters into evangelists rather than skeptics, and lets honest benchmarks ('we lose single-node, we win pooling 70B on consumer GPUs') actually read as confident rather than evasive. Tradeoff: honest status surfaces immaturity to casual readers, but that is strictly better than a defection near the end of a long funnel.

---

### F-217 — [High] Pricing/monetization: the elaborate MARKETING doc fans out into a 7-tier matrix whose linchpin ('billing/') is unimplemented; sequence revenue on the software license + marketplace take-rate, defer the managed cloud

`docs/MARKETING.md:355` · zone=`strategic` · category=`strategic`

- **Summary:** MARKETING.md is a sophisticated open-core plan (Apache 2.0 core, free self-hosted forever, $0.15/GPU-hr PAYG cloud, $50/mo Pro, $1K/node/yr enterprise, 10% Hub take-rate, even a $5/mo Ollama plugin). But its own 'actual implementation is in src/distllm/billing/' — and no such package exists; only a cloud/ compute orchestrator (aws/gcp/azure spot) is present. So today DistLLM has ZERO revenue plumbing. The smart monetization sequence for a single-founder, pre-revenue OSS project is: (1) SELL THE SOFTWARE FIRST at enterprise per-node ($1K/node/yr vs $4,500/GPU/yr NVIDIA NIM is a 5-10x cheaper compliance+support story) because software is the differentiated moat and has near-zero infra cost; (2) MONETIZE THE MARKETPLACE take-rate (10%) as the network-effect line — it is the only part competitors cannot clone; (3) keep PAYG cloud an option but do not invest in a managed cloud you do not yet have, because per-GPU-hour at $0.15 competes head-on at infra cost with Modal/RunPod/Vast.ai where DistLLM has no structural advantage.
- **Evidence (verbatim):**
```
# Simplified model — actual implementation is in src/distllm/billing/ ... (src/distllm/billing/ does not exist in the tree)
```
- **Impact:** Protects the 'most generous open-core' virality claim (MARKETING section 1) from being undercut by a cloud tier users cannot buy, and channels scarce build time to the revenue line (marketplace) with a real moat instead of a me-too GPU-hour price war.
- **Effort:** marketing/doc reorder: 1 week; minimal Stripe metering if pursued: 2-3 weeks
- **Recommendation:** Reorder the pricing deck: enterprise software license and marketplace take-rate are the anchor; managed-cloud PAYG is a later optionality. If a cloud tier is marketed before billing exists, users who try the free '10 GPU-hr/mo' funnel (which drives virality in the doc) will have no way to actually purchase — a credibility gap. Consider shipping a minimal metered Stripe integration (metered GPU-hours, matches the doc's Phase 1) only after the on-ramp and perf are real.
- **Strategic value:** The marketplace take-rate is the asymmetric monetization: it converts the technical moat (reliable heterogeneous GPU aggregation) into a network effect that grows with every GPU hour traded and compounds; per-GPU-hour cloud at $0.15 is structurally undifferentiated against Modal/RunPod/Vast. Tradeoff: two-sided markets are hard to cold-start — hence gating it behind real OSS adoption from moves 1-2 rather than launching it empty in parallel with an unbuilt cloud.

---

### F-218 — [High] Killer feature & real moat = heterogeneous consumer multi-device pooling; all other positioning is cloneable, and the codebase's strength roughly matches the claim

`docs/competitive-analysis.md` · zone=`strategic` · category=`strategic`

- **Summary:** Across README, COMPETITIVE_ANALYSIS, MARKETING, YC_STARTUP_PACKAGE and the actual src/distllm/dist + core surface, the one genuinely defensible capability is: splitting one model across heterogeneous consumer devices (RTX 30/40, AMD, Apple Silicon) over LAN/WiFi/Internet with auto-discovery, node recovery and straggler rebalancing. No one else does this at production grade (Petals is research-grade; Ollama/llama.cpp are single-device; vLLM/DeepSpeed/SGLang assume NVLink/InfiniBand datacenter hardware; Ray Serve is heavyweight, not consumer-NAT-friendly). The dist/ layer evidence this: pipeline/, partition/, recovery.py, straggler.py, rebalancer.py, wide_area.py, discovery.py, nat.py, ice_transport.py, webrtc.py are real, wired code. The defensive moat is NOT the software (a VC-backed Ollama or llama.cpp could clone it) but three compounding assets the project already sketches: (a) the complexity of reliable cross-network pipeline inference, (b) backend-agnosticism hedge (6 backends), (c) a GPU-reputation marketplace network (MARKETING section 2).
- **Evidence (verbatim):**
```
DistLLM is the only product that simultaneously provides: heterogeneous multi-device aggregation ... internet/WAN-capable inference ... auto-discovery and auto-partitioning ... multiple backend support (6 backends) ... zero per-token cost
```
- **Impact:** A crisp, unique wedge is what converts OSS stargazers into installs and later enterprise pilots; scattering positioning across 'fast', 'private', 'cheap', '6 backends' dilutes the one defensible message.
- **Effort:** 1-2 weeks (messaging + roadmap refocus, no code)
- **Recommendation:** Own the 'pool your GPUs' category message explicitly and make the marketing/roadmap narrative singular: multi-device heterogeneity is the wedge; WAN + privacy + marketplace are the fence. Deprioritize anything that pulls the story toward 'another fast inference server' (vLLM/SGLang territory) where the project loses.
- **Strategic value:** Differentiation is assets-based, not feature-count-based. Competing on 'another serving engine' is unwinnable (vLLM/SGLang own datacenter throughput and mindshare); 'pool your own GPUs' is a blue-ocean category with a technical complexity moat. Tradeoff: category creation is slow; it must be paired with an Ollama-adjacent on-ramp to get distribution.

---

### F-219 — [High] Make the OpenAPI spec the single source of truth and gate cross-SDK parity in CI

`sdk/openapi/distllm.yaml:21` · zone=`sdk-arch` · category=`strategic`

- **Summary:** Root cause of most client-layer drift: no authoritative contract. sdk/openapi/distllm.yaml documents only 6 paths but the Python SDK calls ~20 (moderation, audio, images, files, fine_tuning, batch detail/cancel, marketplace, federated, webrtc, defrag). Generated endpoint/type files are stale and unused by the hand-written JS/Go/Rust clients, so generated code, hand-written code, and the server drift apart. Making generation CI-gated fixes contract fidelity, API parity, and streaming parity together.
- **Evidence (verbatim):**
```
spec paths define only chat/completions, completions, embeddings, models, health, batches; Python SDK also POSTs /v1/moderations and /v1/audio/* not in spec
```
- **Impact:** One contract instead of five; new endpoints surface in all SDKs automatically; removes silent per-language divergence.
- **Effort:** 2-4 days
- **Reliability:** Spec lists 6 paths; grep shows SDK POSTs to moderations/audio/images/files/fine_tuning/batch-cancel/marketplace/federated/webrtc/defrag, all absent from spec.
- **Recommendation:** 1) Complete the spec to the real server surface incl. streaming schemas. 2) Commit generate.py to CI and fail on drift (no silent tool-not-found fallback). 3) Wire Go/JS/Rust to generated ENDPOINTS/types instead of re-hardcoding paths. 4) Add a cross-language parity test (identical method coverage + identical stream-item semantics).
- **Strategic value:** A spec-driven, parity-gated multi-SDK story is the differentiator mature platforms (OpenAI/Anthropic/Cloudflare) sell: 'SDKs are trustworthy' vs 'an SDK exists'. Tradeoff: one-time cost to spec ~20 endpoints and pin generator tooling in CI. Reward: JS/Go/Rust stop lagging Python; generated clients become shippable.

---

### F-220 — [High] Rust SDK lacks streaming entirely; JS/Go cover only the core surface

`sdk/rust/src/lib.rs:61` · zone=`sdk-arch` · category=`new_feature`

- **Summary:** Parity is lopsided. Rust exposes only chat_completion, completion, embedding, list_models, health - no streaming. JS covers chat/stream, completions, embeddings, models. Go covers chat/stream, completion, embedding, models, health. None of JS/Go/Rust expose batch, moderation, audio, images, files, fine_tuning, marketplace, or federated, all present in Python. Rust get() also has no retry while post() does.
- **Evidence (verbatim):**
```
rust Client methods are only chat_completion, completion, embedding, list_models, health (lib.rs 61-83); no *_stream method exists anywhere in the rust crate
```
- **Impact:** Rust adopters cannot stream at all; JS/Go users cannot do batch/audio/fine-tuning that Python offers.
- **Effort:** 1-2 days
- **Reliability:** grep of sdk/rust/src/lib.rs and sdk/go/client.go and sdk/js/src/index.ts shows no batch/audio/image/files/fine_tuning methods; rust has no stream method.
- **Recommendation:** Add chat_completion_stream (reqwest SSE) to Rust; port the Python extended surface (batch/audio/images/files/fine_tuning/moderation) to JS/Go/Rust; add retry to Rust get(); gate on the parity test.
- **Strategic value:** Streaming is the core UX of an LLM client; Rust/Go lacking it makes those SDKs non-competitive for the primary use case. Tradeoff: porting effort, mitigated by the single-spec approach producing the types for free.

---

### F-221 — [Medium] Recommmendation: make an honest 'pooling ≠ faster-by-Nx' scaling narrative the product story, not an afterthought

`README.md:27` · zone=`strategic` · category=`strategic`

- **Summary:** The scale-out numbers the project does have (2 nodes = 153 tok/s = 1.66x on 2x GPUs, 83% eff.; 70B figures on the website are 'illustrative, not a BENCHMARKS.md run') are the honest truth of consumer-grade pipeline inference: throughput gains are real for models that don't fit a single card, but scaling efficiency is bounded by chain topology + 1GbE/WiFi. The README Why-table explicitly sells 'Pipeline parallelism = faster generation,' which the data only weakly supports. The durable, unattackable story is: enables models NO single consumer device can host (70B+), with data-locality and near-zero marginal cost — 'faster' should be de-emphasized to avoid a measured claim it can't back at large cluster sizes.
- **Evidence (verbatim):**
```
153.1 tok/s (1.66×, 83.2% eff.) ... 70B numbers 'illustrative, not a BENCHMARKS.md run'; yet README Why-table says 'Pipeline parallelism = faster generation'
```
- **Impact:** Converts the project's biggest apparent weakness (sub-linear scaling, single-node lag) into a controlled, honest strength; removes the most likely source of the 'show HN → dismissive benchmark thread' backlash and the enterprise evaluator's arithmetic that kills deals.
- **Effort:** part of the perf sprint (moves 1/3); copy change 1-2 days
- **Recommendation:** Reframe all customer-facing copy around 'run models you can't otherwise run on the hardware you own, with data that never leaves your devices' and publish a scaling-efficiency curve (tok/s vs node count at 1GbE) so the ceiling is explicit and trusted. Kill the 'faster generation' bullet in README unless multi-node reveals better-than-linear wins that a real BENCHMARKS run can show. The 'runs 70B on consumer GPUs' claim is the one competitors cannot counter.
- **Strategic value:** An honest ceiling invites users whose workload fits inside it, and those are exactly the users who buy (they couldn't run the model at all before). It also decreases the chance of being 'exposed' in a public benchmark comparison against vLLM/SGLang on datacenter hardware, which the docs claim is not their market anyway. Tradeoff: conceding 'not faster at scale' narrows the top-end TAM but is required for long-run credibility.

---

### F-222 — [Medium] Ollama-adjacent on-ramp is the highest-leverage distribution wedge the project has not yet built

`docs/competitive-analysis.md:335` · zone=`strategic` · category=`enhancement`

- **Summary:** COMPETITIVE_ANALYSIS itself identifies the market gap: 'the Ollama multi-device gap — 175K GitHub stars worth of Ollama users who want to cluster their devices' and even prices an 'Ollama Cluster Plugin $5/mo'. Yet no such integration is built (TASKS list none). Because Ollama/llama.cpp are single-device, DistLLM's llama.cpp backend is the perfect layer to give those users a true multi-machine mode with a 'one command to join' experience that mirrors their existing mental model. This is cheap (the distributed machinery already exists) and taps the largest, most conversion-ready audience in the category.
- **Evidence (verbatim):**
```
The Ollama multi-device gap: 175K GitHub stars worth of Ollama users who want to cluster their devices. DistLLM could offer an 'ollama cluster' integration.
```
- **Impact:** Gives the project distribution without having to create a brand-new category from cold; converts the biggest existing local-LLM audience into installs, which in turn populates both the perf win case and, later, marketplace supply.
- **Effort:** 2-4 weeks (reuse existing llama.cpp backend + dist machinery)
- **Recommendation:** Build the drop-in clustering path: preserve Ollama model/pull UX and add a 'join this machine to a cluster' flow backed by the existing llama.cpp backend + auto-discovery + WideArea. Position it explicitly as 'Ollama, but across all your machines,' including the $5-50/mo tier from MARKETING as a low-friction first paid product. Reuse it as the demo story for the HF Spaces launch.
- **Strategic value:** Borrowed distribution beats category creation for a pre-revenue founder. Ollama already solved 'zero-friction local LLM UX'; the differentiating add is the multi-machine layer on top. Tradeoff: piggybacks on Ollama's brand, so the relationship must be framed as additive ('cluster any GGUF across machines') to avoid being read as a fork; also a small surface where Ollama itself could eventually respond by adding clustering.

---

### F-223 — [Medium] CloudArbitrageEngine only generates homogeneous plans — cannot mix heterogeneous instance types to exploit price/compute arbitrage

`src/distllm/dist/partition/cloud_arbitrage.py:399` · zone=`dist-partition` · category=`strategic`

- **Summary:** _generate_candidates always builds candidates from a single instance type repeated num_nodes times; it never mixes e.g. an H100 front stage with cheap L4 tail stages, which is exactly the $/token 'arbitrage' the module's docstring promises. node_ids are also all 'cloud-{i}', so partition/quant heterogeneity across instance types is impossible.
- **Evidence (verbatim):**
```
for instance in catalog: ... for num_nodes in range(1, min(instance.gpu_count+1, 5)):                     nodes = [CloudNode(node_id=f"cloud-{i}", instance=instance, ...) for i in range(num_nodes)]
```
- **Impact:** Missed cost optimization for heterogeneous GPU fleets; the flagship 'cloud arbitrage at $/token' value prop is only partially realized.
- **Effort:** 1-2 days
- **Reliability:** Reading _generate_candidates: the node loop uses a single `instance` variable, so every CloudNode is the same type; no Cartesian/heterogeneous combination code exists.
- **Recommendation:** Generate heterogeneous candidates (e.g., bind different instance types to different pipeline stages, biased by the partitioner's per-node cost), constrain the pairing count, and score by $/token against the same throughput/budget/preemption filters. Strategic value: a genuinely distinct 'pay-for-compute-you-use' placement that streamlines total cost for bursty/spillover workloads.
- **Strategic value:** Market differentiation: true heterogeneous-instance $/token optimization vs single-pool planners; tradeoff is extra combinatorial search cost (mitigate with per-stage instance ranking + top-K pruning) and more complex recovery/checkpointing across mixed instance sizes.

---

### F-224 — [Medium] Byte-level (byte-fallback) tokenizers defeat TokenIndex/JSON first-char mapping used by both constrained decoders

`src\distllm\core\constrained_decoder.py:91` · zone=`core-decoding` · category=`enhancement`

- **Summary:** constrained_decoder.py TokenIndex.build uses `token_str.encode('utf-8')` on `get_vocab()` strings and structured_output JSONSchemaConstraint uses `tokenizer.decode([tid])[0]`. For byte-level BPE tokenizers (LLaMA, GPT-2 byte fallback) the vocab string or decoded first char is `▁`/space or an arbitrary continuation byte rather than the semantically-meaningful first byte, so the first-byte mask forbids the tokens that encode valid JSON ('{', '"', digits) and can block legitimate generation or, conversely, allow tokens whose true byte is illegal. The issue is invisible on non-byte-level tokenizers, so it is a portability/quality hazard rather than always-on.
- **Evidence (verbatim):**
```
for token_str, token_id in vocab.items(): id_to_bytes[token_id] = token_str.encode('utf-8')  # wrong for byte-level BPE vocab reprs
```
- **Impact:** Structured output on byte-fallback models either blocks legitimate tokens (broken/empty output) or lets invalid bytes through, silently depending on the tokenizer family.
- **Reliability:** Code trace: TokenIndex.build get_vocab branch maps vocab string bytes; LLaMA byte-fallback vocab strings are e.g. '▁10', whose first UTF-8 byte (0xe2) is not in any JSON byte set -> all such tokens blocked. JSONSchemaConstraint._build_token_index uses decoded[0] which for byte-fallback is the continuation marker.
- **Recommendation:** Robustify by round-tripping through encode: for each vocab entry call `tokenizer.encode(token_str, add_special_tokens=False)` and use the resulting id, or build id->bytes via `tokenizer.decode([tid])` on the byte stream with `tokenizer.convert_ids_to_tokens(tid)` to resolve byte-fallback (`<0xNN>` tokens). Add a parametrized test over a byte-fallback tokenizer asserting '{'/'"'/digit tokens remain allowed in the right states.

---

### F-225 — [Medium] WebGPU contribution path is decorative: registered browser GPUs never actually execute compute

`src\distllm\core\webgpu_manager.py:356` · zone=`core-gen-rag` · category=`enhancement`

- **Summary:** webgpu_manager.py implements a complete coordinator surface (register_browser / get_available_node / mark_busy / mark_free / heartbeat) but the shipped client HTML's generate() posts directly to /v1/chat/completions (line 356) with stream:false and never routes via /webgpu/inference, never loads web-llm (despite the module docstring promising 'The client uses web-llm for in-browser inference'), and never runs any chunk in the browser. So BrowserGPU.requests_served/total_tokens are never incremented by this client and the 'contribute GPU' story is a stub. register_browser also returns '' on capacity with no client-side handling (line 127).
- **Evidence (verbatim):**
```
await fetch(API_BASE + '/v1/chat/completions', ...)  // not '/webgpu/inference', no web-llm import in HTML
```
- **Impact:** A headline differentiator (zero-install browser GPU contribution) has no executable end-to-end path; the coordinator orchestration is dead code and the browser never contributes compute.
- **Reliability:** Read the module docstring (lines 20-22) vs the single embedded <script> which only does fetch('/v1/chat/completions') — no path sends anything to a registered BrowserGPU.
- **Recommendation:** strategic_value: browser-native GPU pooling is a genuine market differentiator for consumer LAN clusters but is gated on WebGPU maturing in Safari/Firefox; the cost is significant JS/WASM work to actually shard pipeline layers in-browser. A pragmatic increment: implement the /webgpu/inference endpoint that sends a real prefill/gen chunk to a browser worker (web-llm) and streams tokens back, wire the HTML generate() to it, and add a max-capacity error surfaced in the UI — defer full pipeline sharding.

---

### F-226 — [Low] LangChain with_structured_output uses prompt injection instead of the native response_format the SDK already supports

`integrations/langchain/src/distllm_langchain/chat_models.py:367` · zone=`integrations` · category=`enhancement`

- **Summary:** DistLLMChat.with_structured_output (chat_models.py:345-384) appends a 'You MUST respond with a single JSON object conforming to this schema' instruction to the user message and relies on a JsonOutputParser, consuming context budget and offering no guarantee. The SDK's chat_completions already accepts a `response_format` (JSON schema) parameter (sdk/client.py:529) and the platform lists structured output as a feature, so the adapter bypasses the native mechanism.
- **Evidence (verbatim):**
```
instruction = (                 f"\n\nYou MUST respond with a single JSON object that conforms "                 f"to this schema:\n```json\n{schema_str}\n```"             )
```
- **Impact:** Structured output is slower (inflated prompt tokens) and non-deterministic versus the server-side constraint; inconsistent with the framework's native support.
- **Effort:** 2-3 hours
- **Recommendation:** Forward `response_format={"type":"json_schema","json_schema":...}` into the chat_completions payload and parse the returned content, keeping the instruction only as a fallback.
- **Strategic value:** Aligns the adapter with the platform's native structured-output capability (an advertised differentiator) and reduces per-call token cost on multi-draft/structured workloads.

---

### F-227 — [Low] Backend-registry conformance suite: assert every 'available' adapter can actually complete load_model+forward

`src/distllm/backends/registry.py:192` · zone=`backends-config-cloud` · category=`new_feature`

- **Summary:** The zone has 10 registered backends (registry/cross-backend parity thread) with no test asserting protocol conformance: that each is_available()==True backend can construct, load, and produce a (tensor, list) forward without raising NotImplementedError, or that generate() returns a str. Finding 8/9/10 are all instances of this missing gate.
- **Evidence (verbatim):**
```
def list_available(cls): return [p for p in _registry.values() if p.adapter_class.is_available()]
```
- **Impact:** Prevents the recurring class of cross-backend contract regressions this audit surfaced; low cost, high leverage for a 10-backend codebase.
- **Effort:** 1 day
- **Recommendation:** Add tests/backends/test_conformance.py that, for each backend whose deps are installed, runs a mock load (monkeypatch the heavy libs) and asserts forward(input_ids) returns a (Tensor,list) with a documented shape and generate() returns str — turning contract drift (MLX/NIM/WebGPU/TGI/Ollama) into CI failures.
- **Strategic value:** A portable backend conformance harness is the load-bearing spine for the platform's '20+ integrations' promise — it lets maintainers add the 12th backend tomorrow without re-discovering that forward() semantics differ. Tradeoff: needs mock/lib strategy so CI stays light when native deps (vllm/tensorrt/mlx) are absent.

---

### F-228 — [Low] DP RDP accountant: no subsampling-aware RDP and sigma-derived-per-request; consider integration with a maintained privacy library

`src/distllm/core/dp_inference/accounting.py:86` · zone=`core-priv-sec` · category=`enhancement`

- **Summary:** `RDPAccounting.compute_rdp` implements only the non-subsampled Gaussian bound `alpha/(2*sigma^2)` (with an unused `q` param documented but not used) and composes per inference request. Privacy-accounting correctness for arbitrary per-request mechanisms would be more defensible if it delegated to a maintained DP library (opacus/autodp) and recorded the actual mechanism. Currently correctness of the whole chain rests on the hand-rolled advanced-composition formula flagged separately.
- **Evidence (verbatim):**
```
rdp_alpha = alpha / (2.0 * sigma * sigma) rdp_values.append(rdp_alpha * num_queries)  # no sampling-rate q term used
```
- **Impact:** Strengthens confidence in the privacy oracle; reduces risk of subtle accounting drift.
- **Effort:** 1-2 days
- **Reliability:** The `q` sampling-rate docstring/param is not implemented; all accounting assumes q=1 even for batch composition.
- **Recommendation:** Add subsampled-Gaussian RDP (optionally via opacus's rdp) and convert RDP to (eps,delta) with an established routine; wire `num_rounds` composition explicitly. This also lets the under-billing fix be verified against a reference implementation.

---

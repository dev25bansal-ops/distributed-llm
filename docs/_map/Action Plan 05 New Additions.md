---
tags:
  - audit
  - action-plan
date: 2026-08-11
---

# Action Plan 05 — New Additions

**← [[Exhaustive Audit 2026-08-11]]**


14 New Additions for DistLLM, grounded in the 2026-08-11 exhaustive audit digest and a repo survey of the integration seam (integrations/*, sdk/*, apps/*, deploy/*, src/distllm/{api,core,dist,cloud,observability,compliance,backends}). Each addition maps to a documented gap or an existing-but-unwired surface: Prometheus RED metrics are never recorded, MARKETING.md points at a non-existent src/distllm/billing/, the Rust SDK lacks streaming, SSO middleware is unmounted, the K8s operator controller is a logger stub, and the hosted cloud control plane that cloud/worker_agent.py registers to does not exist. Avoids duplicating machinery that already ships (webhook_manager, notification_manager, marketplace.py, daas/, Terraform/Helm, apps/chat, model_registry_router, sso_auth core). Priority mix: P0 x3 (Prometheus /metrics, metered billing, backend conformance+provenance), P1 x6 (Rust SDK parity, OIDC/SCIM SSO, K8s operator, carbon reporting, hosted cloud control plane, structured-output conformance), P2 x5 (outbound alerting notifiers, admin dashboard v2, model-registry governance, framework integrations, community leaderboard+demo). Produced 14 entries spanning ops/observability, monetization/SaaS, SDK parity, enterprise SSO/SCIM, governance/compliance, distribution, and community/ecosystem.

**14 additions**

### N1. Prometheus /metrics scraping endpoint wired to the RED counter/histogram families in observability/exporter.py [priority P0]
- **Description:** Expose a real Prometheus `/metrics` endpoint and populate the metric families that exporter.py already defines with values. The exhaustive audit's Critical finding (api-gateway C1, `src/distllm/observability/exporter.py`) shows RED counters' `.labels()` handles are discarded and never incremented, and a separate finding notes 'exporter.py defines many metrics that nothing ever populates (node_gpu_util, token_latency, cost_per_hour...)'. Add a `prometheus_client`-backed registry in `src/distllm/observability/` that samples usage_meter, request_latency, cache hit-rate, per-route cost/carbon, and tenant SLI, mounted via a new `/metrics` route in server.py, keeping `/metrics_history` (the in-DB JSON variant) intact.
- **Target users:** SRE/platform engineers operating DistLLM clusters; managed-cloud operators; Grafana dashboard consumers; enterprise customers requiring Prometheus-onboarding.
- **Value:** Delivers the observability the marketing promises (acceptance-rate, cache hit-rate, per-route cost/carbon, tenant SLO). Turns the audit's most-severe metrics bug into a shipped capability; enables Grafana dashboards, alerting on error-rate/burn-rate, and truthful capacity/cost reporting. Highest-value observability addition because it unblocks SLA/uptime SLOs (7.3: 'external SLOs 99.9% API availability').
- **Integration points:** src/distllm/observability/exporter.py and metrics.py (metric definitions exist); src/distllm/api/server.py (add `app.add_route('/metrics', ...)` alongside metrics_history_router at server.py:927); core/usage_meter.py, core/request_latency.py, core/cache_manager.py as data sources; dist/metrics.py + dist/otel.py for cross-node aggregation.
- **Alignment:** S4 'switchable inference broker' and 7.3 reliability/SLO; the Core Audit 05 N7 Prometheus contract; strengthens the commercial serving-layer story (billing/quota).
- **Effort:** 3-5 engineer-days
- **Evidence:** audit_digest C1 (RED metrics .labels() discarded); digest 'Medium core-perf-obs' exporter.py metrics never populated; src/distllm/observability/exporter.py, metrics.py; src/distllm/api/server.py:927 (metrics_history_router only).

---

### N2. src/distllm/billing/ package: Stripe + Chargebee metered-billing webhooks, /v1/billing endpoints, self-serve portal, reconciliation [priority P0]
- **Description:** Build the billing layer MARKETING.md claims shipped but that does not exist. docs/MARKETING.md:355 asserts 'actual implementation is in src/distllm/billing/' — that directory is absent from the repo. MARKETING.md:309-326 specify Stripe (credited usage) + Chargebee/Metronome for enterprise. Create `src/distllm/billing/` with: (1) a MeteredUsageReporter that syncs hourly GPU-seconds from usage_meter to Stripe metered-billing API; (2) webhook handlers (invoice.paid, customer.subscription.updated, dunning retries) mounted in server.py; (3) `/v1/billing` endpoints (plan, usage, invoice list, portal session); (4) a reconciliation check that flags usage_meter under-reporting (audit: 'UsageMeter time-window queries under-report after 100k records'). Also fix the monthly cost-budget comparison bug ('Monthly cost-budget enforcement compares against all-time total_cost').
- **Target users:** Managed-cloud SaaS operator; individual devs on pay-as-you-go tier; enterprise accounts.
- **Value:** Turns the flagship monetization from marketing-only into shippable software; enables per-token charging; reconciles the usage-meter under-billing audit bug.
- **Integration points:** src/distllm/core/usage_meter.py + tenant_billing.py (metering source); src/distllm/core/metering.py (hooks); src/distllm/api/server.py (mount routers); new src/distllm/billing/{stripe,chargebee,reconcile}.py; dashboard for portal links. Reconcile against audit finding: usage_meter under-reports after 100k records.
- **Alignment:** S4 revenue path; 'pricing/monetization ... linchpin unimplemented' strategic finding; commercial-serving-layer moat (billing/tenancy is a differentiator).
- **Effort:** 3-6 engineer-weeks for a production path
- **Evidence:** docs/MARKETING.md:355 (points at absent src/distllm/billing/), :268-326; audit 'High strategic' pricing/monetization unimplemented; 'Month cost-budget enforcement compares against all-time total_cost' (core-perf-obs); usage_meter under-report (dist-exec).

---

### N3. Rust SDK streaming + embeddings/tools/structured-output surface; cross-SDK parity gate [priority P1]
- **Description:** Bring the Rust SDK (and full JS/Go surface) to parity with the Python SDK. Audit finding (sdk-arch High, new_feature) 'Rust SDK lacks streaming entirely; JS/Go cover only the core surface'. Confirmed: sdk/rust/src/lib.rs has a single 'stream' mention; sdk/go/client.go:232-234 has ChatCompletionStream (returns <-chan string, <-chan error) but no embeddings/tools; sdk/js/index.ts:222 has async* stream but limited chat/completions. Add: Rust `stream_completion` SSE parser with token deltas, embeddings, function/tool-calling (bind_tools), and structured-output response_format; document the cross-SDK parity matrix in sdk/openapi/distllm.yaml. Gate parity in CI so the OpenAI-compat surface is identical across python/go/js/rust.
- **Target users:** Rust/Go/JS/edge developers; Tauri desktop clients.
- **Value:** Streaming + parity across all four SDKs; removes a High-priority gap and strengthens the multi-language promise.
- **Integration points:** sdk/rust/src/lib.rs + sdk/rust/src/generated/*; sdk/js/src/index.ts; sdk/go/client.go; sdk/openapi/distllm.yaml (spec-of-truth); sdk/tests for parity; Tauri apps (sdk-arch note 'Tauri chat calls REST without auth').
- **Alignment:** sdk-arch parity finding; OpenAI-compat breadth as a differentiator; increases developer radar.
- **Effort:** 1-2 engineer-weeks (Rust streaming is the bulk)
- **Evidence:** audit 'sdk-arch High new_feature' Rust SDK no streaming; sdk/rust/src/lib.rs (1 'stream'); sdk/go/client.go:232-234; sdk/js/src/index.ts:222,271; audit 'sdk-arch High strategic' OpenAPI as source of truth.

---

### N4. OIDC + SAML SSO provider registry and SCIM user/group provisioning, mounting the unwired sso_middleware [priority P1]
- **Description:** Productize the documented-but-unmounted SSO flow. Audit finding (I3 / strategic) shows sso_middleware.py (626 LOC) defines SSO middleware + `/v1/auth/*` endpoints but is NOT wired into server.py — the endpoints it defines are unmounted; only sso_auth.py core + api/auth/* handlers are live. Add: an SSO provider registry (Okta, AzureAD/Entra, Google, GitHub, OIDC-generic, SAML) implementing a full OIDC Authorization Code + PKCE and SAML AssertionConsumer flow behind the existing tls; `/v1/auth/authorize` + `/v1/auth/callback`; SCIM 2.0 endpoints for /Users and /Groups so IdPs can provision/update/deactivate tenant users and roles; wire into api/authz/opa.py for RBAC.
- **Target users:** Enterprise SSO/SCIM customers; regulated industries.
- **Value:** Mounts an existing 626-LOC SSO implementation; adds OIDC/SAML/SCIM to reach enterprise buyers.
- **Integration points:** src/distllm/api/sso_middleware.py (mount it in server.py), src/distllm/api/sso_auth.py, api/auth/* (auth/token handlers), api/authz/opa.py (RBAC), api/persistent_store.py (session store); new api/scim/routes.py.
- **Alignment:** I3 SSO-unwired strategic fix; enterprise/GDPR narratives; single-auth-decision-point enhancement.
- **Effort:** 1-2 engineer-weeks (OIDC) + SCIM 1 week
- **Evidence:** audit 'I3' / strategic — sso_middleware unwired; src/distllm/api/sso_middleware.py (626 LOC); only sso_auth.py + auth/* live per the strategic doc.

---

### N5. Outbound incident notifier sinks (Slack, Discord, PagerDuty, email, webhook) over webhook_manager/notification_manager + incident_response [priority P2]
- **Description:** Add concrete delivery sinks for the existing alerting/notification machinery. src/distllm/core/webhook_manager.py and notification_manager.py exist but there is no outbound Slack/Discord/PagerDuty/email sink, and observability/incident_response.py + alerting_config.py define alert events. Add `src/distllm/observability/notifiers/` with a pluggable Notifier that: (a) formats SLO burn-rate alerts from slo_config, (b) routes to Slack/Discord webhooks, PagerDuty Events API v2, SMTP/Resend, or a generic webhook, (c) integrates with incident_response to open/update/resolve incidents, (d) adds dedup + escalation policy by severity.
- **Target users:** Ops/SRE teams and on-call responders.
- **Value:** Actionable, dedup'd on-call alerts from SLO/metrics; reuses existing managers.
- **Integration points:** src/distllm/core/webhook_manager.py (webhook_definitions seam), core/notification_manager.py, observability/incident_response.py + alerting_config.py + slo_config.py (event sources); new observability/notifiers/; dashboard/ws_handler for in-UI alerts.
- **Alignment:** 7.3 reliability & SLO.
- **Effort:** 3-5 engineer-days
- **Evidence:** src/distllm/core/webhook_manager.py, core/notification_manager.py, observability/incident_response.py, alerting_config.py, slo_config.py all exist; no Slack/Discord/PagerDuty sink anywhere.

---

### N6. Kubernetes operator GA: real reconcile loop + DVCRD + autoscale hook over deploy/operator/controller.py [priority P1]
- **Description:** Replace the placeholder operator controller with an actual controllers. deploy/operator/controller.py currently only does logger.info and makes no K8s API calls (strategic finding), while deploy/crds + deploy/helm/distllm-operator + deploy/kustomize scaffold the plumbing. Add: (1) a DistLLMCluster custom-resource Controller with a reconcile loop (create workers, wire gRPC/TLS, propagate image/config), (2) readiness/liveness probes and status subresources, (3) K8s HPA + KEDA hook that actuates the intelligent autoscaler (audit: 'IntelligentAutoscaler wired but never fed or actuated'), (4) a ClusterRole/RoleBinding manifest and RBAC tests.
- **Target users:** Enterprise platform teams on Kubernetes.
- **Value:** Real reconcile-loop operator, the stated enterprise adoption gate; wires the never-actuated autoscaler.
- **Integration points:** deploy/operator/controller.py (rewrite), deploy/crds/ (CRD spec), deploy/helm/distllm-operator/ + deploy/kustomize/ (charts), src/distllm/core/aria_autoscaler.py / IntelligentAutoscaler (scale actuation), api/health for probes.
- **Alignment:** N8; enterprise adoption.
- **Effort:** 3-4 engineer-weeks
- **Evidence:** audit finding — deploy/operator/controller.py logger.info only, no K8s API calls; deploy/crds, deploy/helm/distllm-operator, deploy/kustomize scaffold present; IntelligentAutoscaler wired-but-never-fed (core-ops-ha).

---

### N7. Backend-registry conformance gate + signed-model provenance and SBOM export [priority P0]
- **Description:** Add a production integrity layer the audit flags as missing. Audit 'new_feature' Low (backends-config-cloud) asks for a 'Backend-registry conformance suite: assert every available adapter can actually complete load_model+forward' (many adapters advertise available but fail: WebGPU is_available==True yet forward raises NotImplementedError; NIM fabricates logits; Azure always fails open). Add: (1) a conformance suite `tests/backends/conformance.py` that runs load_model+forward on every available adapter (skip-unavailable), gated in CI; (2) an integrity gate before `load_model` that verifies model provenance (HF Hub download hash / signed manifest) and rejects untrusted `trust_remote_code`; (3) `distllm provenance export` writing an SBOM (SPDX) of all loaded models + their source hashes.
- **Target users:** Platform/model-safety teams; regulated enterprises.
- **Value:** Conformance catches broken backends; model provenance + SBOM meets supply-chain and export-control needs.
- **Integration points:** src/distllm/backends/registry.py (conformance + load_model gate), src/distllm/backends/protocol.py (BackendAdapter contract), new src/distllm/security/model_provenance.py + core/verification/; integrate with security/edge_attestation or verifier; CLI distllm/verification.
- **Alignment:** trust-first product; conformance new_feature.
- **Effort:** 1-2 engineer-weeks
- **Evidence:** audit findings: adapter conformance suite (backends-config-cloud new_feature); WebGPU available-but-notImplemented; NIM fabricated logits; Azure fails open; MODEL_COMPATIBILITY.md, EXPORT_CONTROLS.md.

---

### N8. Carbon & GHG reporting export sink (CSV/XLSX/CSRD-format) over energy/carbon_migration/cross_cloud_router [priority P1]
- **Description:** Turn the existing carbon/energy modules into a reportable product. core has advanced_scheduling/energy.py, carbon_migration.py, cross_cloud_router.py, pareto_optimizer.py but no way to export a GHG/CSRD audit. Add `src/distllm/compliance/carbon_report.py`: (1) a per-tenant, per-route, per-model emissions ledger pulled from the energy/carbon wiring, (2) export to CSV/XLSX and a CSRD/ESRS-aligned JSON schema, (3) a `/v1/carbon/report` API endpoint (audit-gated) returning the signed report, (4) reconcile against cost_dashboard so carbon+$ are comparable.
- **Target users:** Enterprise ESG/CSRD teams; regulated companies.
- **Value:** Turns carbon-aware routing into a reportable enterprise product.
- **Integration points:** src/distllm/core/advanced_scheduling/energy.py + core/carbon_migration.py + core/cross_cloud_router.py + core/pareto_optimizer.py (data); src/distllm/compliance/evidence_pack.py (signing); docs/GDPR.md + EXPORT_CONTROLS.md; api server add /v1/carbon/report; dashboard cost_dashboard.
- **Alignment:** S2/F3 carbon wedge; governance/compliance.
- **Effort:** 1-2 engineer-weeks
- **Evidence:** core/advanced_scheduling/energy.py, core/carbon_migration.py, core/cross_cloud_router.py, core/pareto_optimizer.py exist but no reporting/export; compliance/evidence_pack.py exists for attestation.

---

### N9. Admin dashboard v2 GA: usage/cost/SLO/tenant-management panels over dashboard static_v2 + usage_meter [priority P2]
- **Description:** Productionize the v2 dashboard into the operator/admin console. Audit (I12) finds dashboard v1 and v2 duplicated (dashboard/static vs static_v2) and dashboard/ui empty; productionize v2. Add panels wired to real data: usage_meter (tokens/GPU-hours per tenant), tenant_billing / tenant_cost_attribution (cost), usage SLA tier panels, and tenant management (quota, RBAC, disable). Mount as `/dashboard/v2` behind admin auth; include the metrics WebSocket already present (server.py:970, :1032).
- **Target users:** Cluster admins; managed-cloud operators.
- **Value:** Operator/tenant console over real usage/cost/SLO data; kills v1/v2 duplication.
- **Integration points:** src/distllm/dashboard/static_v2 (GA) + src/distllm/dashboard/ws_handler.py (live metrics WS server.py:970-1032); core/usage_meter.py, core/tenant_billing.py, core/cost_dashboard.py, core/tenant_cost_attribution.py; api/admin.py router; authz/opa.py RBAC.
- **Alignment:** I12; commercial serving layer.
- **Effort:** 2-3 engineer-weeks
- **Evidence:** src/distllm/dashboard/ has static, static_v2, ui dir (ui empty); server.py:970/:1032 WebSocket metrics; usage_meter.py, cost_dashboard.py, tenant_cost_attribution.py all exist.

---

### N10. Model-registry governance layer: approval workflow, license/use policy, content-moderation + DPI redaction per model [priority P2]
- **Description:** Add a governance/approval layer over the existing model_registry_router (mounted at server.py:923). New models enter with statuses pending/approved/blocked; admins approve after policy checks (license allowlist, export-control flag from docs/EXPORT_CONTROLS.md, content-moderation policy from src/distllm/security/content_moderation). Apply per-model DP/inference redaction (core/privacy_budget, core/dp_inference/accounting) and per-tenant quotas. Expose `/v1/models/{id}/approve` and a policy engine wired to api/authz/opa.py.
- **Target users:** Governance officers; regulated tenants; marketplace curators.
- **Value:** Approval + policy + DPI/redaction per model; meets governance/compliance needs.
- **Integration points:** src/distllm/api/routes/model_registry.py (router), security/content_moderation, core/privacy_budget + core/dp_inference/accounting, api/authz/opa.py (policy), compliance/evidence_pack.py; export-control flags from docs/EXPORT_CONTROLS.md.
- **Alignment:** compliance/governance; trust-first.
- **Effort:** 1-2 engineer-weeks
- **Evidence:** server.py:923 model_registry_router mounted; src/distllm/security/content_moderation/ exists; dp_inference/accounting.py + privacy_budget.py + compliance/evidence_pack.py exist; docs/EXPORT_CONTROLS.md + GDPR.md.

---

### N11. Hosted 'DistLLM Cloud' control-plane reference service (daas_server) that cloud/worker_agent.py can actually register to [priority P1]
- **Description:** Provide the control-plane endpoint the worker join code already targets. src/distllm/cloud/worker_agent.py implements register_worker that POSTs capability metadata to a /register endpoint, but the strategic finding (E13) says it 'joins a non-existent control plane' — the hosted coordinator does not exist. Add a reference `daas_server` control plane (mount in server.py): /v1/cloud/register, /v1/cloud/heartbeat, worker_identity provisioning; plus a `distllm cloud join <url>` CLI and a Cloud-choose orchestrator that feeds cross_cloud_router / spot_orchestrator. This is the single biggest untapped asset — worker join code already exists.
- **Target users:** Self-hosters; managed-cloud operator; spare-GPU contributors.
- **Value:** The control plane workers already register to; unlocks the hosted product and capacity marketplace.
- **Integration points:** src/distllm/cloud/worker_agent.py register_worker (existing); src/distllm/dist/daas_server.py + dist/daas/ (tenant_dispatcher, load_balancer, marketplace_integration); api server mount cloud_control router; cross_cloud_router.py + spot_orchestrator.py as backends; cli/main.py add 'cloud join'.
- **Alignment:** N1/S1; distribution moat.
- **Effort:** 4-6 engineer-weeks
- **Evidence:** src/distllm/cloud/worker_agent.py:7-9 ('register with a hosted cloud coordinator so ... DistLLM Cloud hosts only the control plane'), :59, :137; dist/daas_server.py + daas/ exist; audit E13 'worker_agent joins non-existent control plane'.

---

### N12. New framework integrations: PydanticAI (Python), Vercel AI SDK (JS/TS), DSPy; + OpenAI-Agents/Semantic-Kernel streaming+tool parity fixes [priority P2]
- **Description:** Extend the 28-adapter ecosystem to the fastest-growing agent frameworks, and fix the parity bugs the audit found in the strongest adapters. Audit (High, integrations) 'Streaming is fully broken across langchain/crewai/llamaindex adapters (TypeError on stream=True...)' and 'Tool-calling contract broken: bind_tools() injects tools/federation kwargs the SDK does not accept'. Add: (1) a PydanticAI Integration (tools + structured output, leveraging SDK native response_format); (2) a Vercel AI SDK (JS) provider implementing LanguageModelV2 streamText/toolCall; (3) a DSPy Retriever/Generator; (4) fix the streaming type and tool-kwargs contract across langchain/crewai/llamaindex/agno so the existing adapters actually work.
- **Target users:** Python/JS agent developers; enterprise adopters.
- **Value:** Adds PydanticAI/Vercel/DSPy adapters; fixes the High streaming+tool-contract bugs in existing adapters.
- **Integration points:** integrations/ dir (add pydanticai/, vercel-ai-sdk/, dspy/); fix integrations/langchain, crewai, llamaindex adapters (stream + bind_tools); sdk/src/distllm_sdk/compat for response_format/structured output; docs/EXAMPLES_APPS_NOTEBOOKS.md.
- **Alignment:** developer radar; ecosystem.
- **Effort:** 1-2 engineer-weeks
- **Evidence:** audit integrations High streaming broken across langchain/crewai/llamaindex; tool-calling contract broken; src/distllm/api/routes/tools.py; integrations/langchain/src/distllm_langchain/chat_models.py (with_structured_output uses prompt injection per audit Low enhancement).

---

### N13. Structured-output conformance suite: JSON-Schema->GBNF grammar interop + multi-digit-number FSM fuzzer [priority P1]
- **Description:** Harden the constrained-generation layer with a conformance + fuzz suite. Audit findings: JSONSchemaConstraint FSM 'never allows continuation of multi-digit numbers — constrained generation truncates' (High core-gen-rag); GBNF grammar emits invalid 'value' rule for $ref/$defs (ops-utils); two parallel structured-output stacks (core OutputRepairer vs dist RepairOrchestrator). Add: (1) a conformance harness that samples generation under JSON-Schema constraints and validates against jsonschema, covering multi-digit numbers, nested objects, $ref/$defs; (2) a fuzzer generating random valid JSON-Schemas and asserting constraint coverage; (3) unify core/dist structured-output repair onto one API.
- **Target users:** Engineers using structured output / response_format.
- **Value:** Conformance + fuzz gate over constrained decoding; fixes the multi-digit truncation bug.
- **Integration points:** src/distllm/core/structured_output/ + core/constrained_decoder.py (fix FSM); src/distllm/dist/structured_output/ + RepairOrchestrator vs core OutputRepairer (consolidate); tests/fuzz/ add json-schema fuzzer; sdk/openapi/distllm.yaml response_format surface.
- **Alignment:** correctness-orientation; structured-output feature.
- **Effort:** 3-5 engineer-days
- **Evidence:** audit High core-gen-rag JSONSchemaConstraint multi-digit truncation; ops-utils GBNF invalid 'value' rule; 'two independent structured-output repair stacks' (core-gen-rag); sdk/rust lacking structured output (parity).

---

### N14. Community assets: public eval-leaderboard publisher + Streamlit/Gradio demo+evals bundle in apps/ [priority P2]
- **Description:** Add developer-relations assets that convert the benchmark work into community proof. docs/BENCHMARKS.md + benchmarks/ competitive_benchmark exist but there is no public leaderboard publishing or a turnkey demo app. Add: (1) `distllm leaderboard publish` that formats a benchmark run into a shareable HTML/JSON card and posts to an OSS leaderboard page (reuse leaderboard_router mounted at server.py:921); (2) a `apps/evals_demo` Streamlit/Gradio bundle with a chat UI + a pre-built eval/latency showcase wired to the running server, one command to launch.
- **Target users:** Community, prospective customers, contributors.
- **Value:** Public leaderboard + one-command demo app convert benchmark work into adoption proof.
- **Integration points:** apps/ append evals_demo; api/routes/leaderboard.py (leaderboard_router at server.py:921); benchmarks/competitive_benchmark.py as source; docs/BENCHMARKS.md + PERFORMANCE_COMPARISON.md to link the public leaderboard; cli/main.py add 'leaderboard publish'.
- **Alignment:** S5; community & adoption.
- **Effort:** 3-5 engineer-days
- **Evidence:** leaderboard_router mounted at server.py:921; docs/BENCHMARKS.md + PERFORMANCE_COMPARISON.md exist; apps/ has chat, multi_agent, rag but no evals/demo bundle; strategic S5 benchmark-led credibility.

---

# Monetization Strategy

Design across 9 dimensions for DistLLM's open-core monetization.

---

## 1. Open-Core Free vs Paid Split

### Open-Source Core (Apache 2.0) — Free

The core is genuinely useful standalone. A user can run production inference on owned hardware indefinitely without ever paying DistLLM.

| Module | Free | Paid |
|--------|------|------|
| Pipeline parallelism engine | ALL | -- |
| All 6 backends (vLLM, llama.cpp, TRT-LLM, ExLlamaV2, ONNX, PyTorch) | ALL | -- |
| CLI tooling (`distllm cluster start/join`) | ALL | -- |
| Auto-discovery (mDNS/zeroconf) | ALL | -- |
| Auto-partitioning by device VRAM | ALL | -- |
| OpenAI-compatible API | ALL | -- |
| PyPI package (`pip install distllm`) | ALL | -- |
| SDKs (Python, TypeScript, Go) | ALL | -- |
| Observability (Prometheus metrics, basic dashboard) | ALL | -- |
| Semantic caching | ALL | -- |
| Circuit breaker system | ALL | -- |
| LAN mode | ALL | -- |
| Node recovery (checkpoint-based) | ALL | -- |
| Basic straggler detection | ALL | -- |

### Paid Features

Only enterprise/compliance/scale features are gated. No core inference capability is withheld.

| Feature | Product | Rationale |
|---------|---------|-----------|
| SSO/SAML/LDAP | Enterprise | Enterprise procurement requirement, low usage in self-serve |
| RBAC with audit logging | Enterprise | Compliance mandate |
| SOC 2 Type II compliance | Enterprise | $50K+ audit cost must be monetized |
| HIPAA compliance & BAA | Enterprise | Legal liability, niche need |
| Air-gapped deployment | Enterprise | Enterprise procurement requirement |
| WAN-optimized mode (NAT traversal) | Enterprise | Power-user feature, not core to most users |
| Dynamic rebalancing (live straggler mitigation) | Enterprise | Advanced reliability, mostly needed in production at scale |
| Multi-cluster management dashboard | Enterprise | Operations at scale |
| Premium support (4hr SLA) | Enterprise | Requires dedicated headcount |
| Managed cloud clusters | Cloud | Infrastructure cost offset |
| Priority scheduling / faster queuing | Cloud Pro | Convenience |
| GPU reputation marketplace | Hub (separate) | Marketplace dynamics |
| Federated fine-tuning | Enterprise (future) | Niche, complex |
| Ollama Cluster Plugin | Paid standalone | Adjacent product, different audience |

### Guiding Principle

> Never gate a feature that would make a user's first-run experience worse.
> The open-core must be a complete, production-quality product on its own.
> Paid features solve problems users only encounter at scale or under compliance.

---

## 2. Marketplace Fee Model

### DistLLM Hub — GPU Reputation Marketplace

A two-sided marketplace connecting GPU owners ("providers") with users who need inference compute ("renters").

### Fee Structure

| Component | Detail |
|-----------|--------|
| **Take rate** | 10% on all transactions |
| **Provider payout** | 85% of transaction (DistLLM retains 10% + 5% for payment processing/escrow) |
| **Listing fee** | $0 (free to list) |
| **Minimum transaction** | $1.00 |

### How It Works

1. Provider installs DistLLM Hub agent on their machine
2. Agent runs attestation proofs (actual GPU capacity, uptime benchmark, latency test)
3. Provider sets hourly rate (DistLLM suggests $0.05-0.15/GPU-hour based on GPU type)
4. Renter discovers providers ranked by reputation score, price, latency, and capacity
5. Renter pays per GPU-hour via DistLLM Hub (escrow held)
6. DistLLM takes 10% cut at settlement
7. Both parties rate each other (reputation system)

### Reputation System

- Providers earn reputation through verified benchmarks, completed jobs, uptime, low latency
- Renters earn reputation through payment history, job completion rate, provider ratings
- Higher-reputation providers appear first in search results
- Low-reputation providers require escrow holding (longer settlement)
- Malicious behavior (fake GPU claims, service fraud) leads to permanent ban

### Revenue Estimate

At 1M GPU-hours/year traded: 1M × $0.10 avg × 10% = $10K revenue/year (Year 2-3 target).

---

## 3. Enterprise Tier Features & Pricing

### Enterprise Self-Hosted

**Price: $1,000/node/year (minimum 2 nodes)**

Volume discounts:
| Volume | Price/node/year | Discount |
|--------|----------------|----------|
| 1-4 nodes | $1,000 | — |
| 5-9 nodes | $850 | 15% |
| 10-49 nodes | $700 | 30% |
| 50+ nodes | $500 | 50% |

### Included Features

| Category | Features |
|----------|----------|
| **All open-source** | Pipeline parallelism, 6 backends, CLI, SDKs, auto-discovery, semantic caching, circuit breakers, node recovery |
| **Identity & Access** | SSO/SAML/LDAP, SCIM provisioning, role-based access control (RBAC), API key management with rotation policies |
| **Compliance** | SOC 2 Type II report access, HIPAA Business Associate Agreement (BAA), GDPR data processing agreement, compliance audit logs (immutable), data residency controls |
| **Deployment** | Air-gapped installation, Helm charts for Kubernetes, Terraform modules, offline license validation |
| **Operations** | Multi-cluster management dashboard, alerting integrations (PagerDuty, Slack, OpsGenie), usage analytics and cost allocation, health checks and auto-remediation |
| **Support** | Dedicated Slack channel, 4-hour response SLA (business hours), 24-hour response SLA (nights/weekends), quarterly business review, named support engineer |
| **Security** | Audit logging (immutable, tamper-evident), encryption key management (BYOK), vulnerability scanning reports, penetration test results (annual) |

### Enterprise Cloud (Managed, Same Compliance)

**Price: Custom quoting — typically $2,000-5,000/month**

For organizations that want managed infrastructure but enterprise compliance.

### Enterprise Annual Contract Terms

- Annual commitment with quarterly true-ups
- Net-30 invoicing or PO
- Optional: dedicated GPU reservation at published rates minus enterprise discount
- Optional: model fine-tuning pipeline (distilled, privacy-preserving) — add-on at $500/month

---

## 4. Managed Cloud Pricing

### DistLLM Cloud

Two tiers plus custom:

#### Cloud Pay-as-you-go

| Component | Price |
|-----------|-------|
| **GPU compute** | $0.15/GPU-hour (blended across GPU types) |
| **Storage** | $0.10/GB/month (model cache, KV cache persistence) |
| **Networking** | Included (up to 1TB egress/month, then $0.05/GB) |
| **Billing granularity** | Per-second with 60-second minimum |
| **Free monthly allowance** | 10 GPU-hours/month (resets monthly) |

Typical monthly cost (70B model, sustained): ~$100-350/month vs $600-8,500+ for cloud API alternatives.

#### Cloud Pro

| Component | Price |
|-----------|-------|
| **Monthly base** | $50/month |
| **GPU compute** | $0.12/GPU-hour (20% discount over pay-as-you-go) |
| **Storage** | $0.08/GB/month |
| **Networking** | 5TB egress included |
| **Perk** | Priority scheduling queue, faster cold-start, 2x rate limits |
| **Support** | Email support with 8-hour response SLA |

#### Commitment Discounts (any tier)

| Commitment | Discount | Effective GPU-hour price |
|------------|----------|-------------------------|
| 1-month commit (min $200) | 10% | $0.135 |
| 3-month commit (min $500) | 20% | $0.12 |
| Annual commit (min $2,000) | 30% | $0.105 |
| Spot/preemptible (no commit) | 50% off on-demand | $0.075 |

### Unit Economics

| Metric | Value |
|--------|-------|
| Blended revenue per GPU-hour | $0.15 |
| Compute cost per GPU-hour | $0.06-0.09 (GPU instance + networking) |
| Gross margin | 40-60% |
| Target break-even on Cloud | 500 paying GPU-hours/day (~$22.5K MRR at blended rate) |
| Path to 70%+ margin | Reserved instances, spot instances, larger commitment volumes |

---

## 5. Tier Structure

### Summary Table

| Tier | Price | Best For | Key Features | Support | SLA |
|------|-------|----------|-------------|---------|-----|
| **Free (OSS)** | $0 | Hobbyists, students, researchers | Full open-core, LAN, 6 backends, auto-discovery, community | Discord, GitHub Issues | Best-effort |
| **Cloud Pay-as-you-go** | $0.15/GPU-hr (first 10 hrs free/mo) | Individual devs, indie builders | Managed clusters, 2-min setup, per-second billing, OpenAI API | Discord + email | 99.5% uptime |
| **Cloud Pro** | $50/mo + $0.12/GPU-hr | Professional devs, small teams | Priority scheduling, faster queues, 2x rate limits, enhanced support | Email (8hr SLA) | 99.5% uptime |
| **Enterprise Self-Hosted** | $1K/node/yr (min 2) | Regulated orgs, mid-market | SSO, RBAC, SOC 2, HIPAA, air-gap, audit logs, multi-cluster mgmt | Slack + dedicated engineer | 99.9% uptime |
| **Enterprise Cloud** | Custom (~$2-5K/mo) | Enterprises wanting managed + compliance | Same features as Self-Hosted but managed by DistLLM | Slack + dedicated engineer | 99.9% uptime |
| **Hub Marketplace** | 10% take rate | GPU owners / compute renters | Peer-to-peer GPU sharing, reputation, escrow | Discord | Best-effort |
| **Ollama Plugin** | $5/mo (individual), $50/mo (team) | Ollama users wanting multi-device | One-command cluster plugin for Ollama | Email | Best-effort |

### Upgrade Path

```
                  ┌─────────────────┐
                  │  Free OSS       │  ← GitHub, HN, Reddit, word of mouth
                  │  (self-hosted)  │
                  └────────┬────────┘
                           │ Need more GPUs / don't want ops
                           ▼
                  ┌─────────────────┐
                  │  Cloud Pay-as-  │  ← First 10 hrs free / credit card
                  │  you-go         │
                  └────────┬────────┘
                           │ Need priority / more throughput
                           ▼
                  ┌─────────────────┐
                  │  Cloud Pro      │  ← Self-serve upgrade
                  │  ($50/mo)       │
                  └────────┬────────┘
                           │ Need compliance / SSO / air-gap
                           ▼
                  ┌─────────────────┐
                  │  Enterprise     │  ← Sales-led
                  │  ($1K/node/yr)  │
                  └─────────────────┘
```

---

## 6. Pricing Comparison vs GitLab / HashiCorp / Supabase

| Dimension | GitLab | HashiCorp | Supabase | **DistLLM** |
|-----------|--------|-----------|----------|-------------|
| **OSS license** | MIT (Core) | BSL (moved to BSL + commercial) | Apache 2.0 | **Apache 2.0** |
| **Open-core generosity** | Moderate — CE is usable but many useful features (merge approvals, security scanning) are EE-only | Good — Terraform core is complete, HCP adds convenience and compliance | Excellent — free tier is genuinely useful, only scale/compliance gated | **Excellent — core is fully production-capable standalone** |
| **Free tier** | GitLab.com Free (400 CI min/mo, 5 users) | HCP free (5 users, limited resources) | Free (2 projects, 500MB DB, 50K rows) | **Self-hosted: unlimited. Cloud: 10 GPU-hr/mo** |
| **Entry paid** | $19/user/mo (Premium) | $20/seat/mo (HCP Standard) | $25/mo (Pro) | **$0.15/GPU-hr (pay-as-you-go)** |
| **Team/Pro tier** | $19/user/mo | $20/seat/mo | $25/mo | **$50/mo + usage** |
| **Enterprise** | $99/user/mo (Ultimate) | $50-160/seat/mo (Enterprise tiers) | $25K-50K+/yr (Custom) | **$1K/node/yr (self-hosted)** |
| **Pricing model** | Per-seat (user-based) | Per-seat (user-based) | Usage + seat hybrid | **Usage-based (GPU-hour) + per-node** |
| **Primary monetization** | Seat licenses | Seat licenses + HCP consumption | Pro/Team subscriptions + Compute Credits | **Cloud GPU-hour + Enterprise node license** |
| **Self-serve ceiling** | ~$200/mo (then sales) | ~$500/mo | ~$1,000/mo | **~$500/mo (then sales-assisted)** |
| **SOM at IPO/acq** | $500M+ ARR | $1B+ ARR | $100M+ ARR | **Target $50M ARR (Year 4)** |

### DistLLM Differentiation

1. **Most generous open-core of the four**: Apache 2.0 + genuinely complete free product. You can run 70B models across your LAN indefinitely at zero cost. This drives maximum adoption.

2. **Usage-based pricing (not per-seat)**: Per-seat pricing punishes the most successful users (who add more users). GPU-hour pricing aligns with value delivered — you pay for compute used, not headcount. This works because GPU compute is the actual scarce resource.

3. **Lower absolute spend than cloud API alternatives**: $100-350/month for sustained 70B workloads vs $600-8,500+. The pricing is a fraction of cloud API pricing while still generating 40-60% margin.

4. **Self-hosted enterprise is the anchor**: $1,000/node/year is a fraction of what a single GPU costs (a single RTX 4090 costs $1,600-2,000 upfront). For an enterprise with 10 nodes, $7,000-10,000/year for compliance and support is a small fraction of the hardware + power cost.

---

## 7. Self-Serve vs Sales-Led Model

### Hybrid Approach

#### Self-Serve (Phases 1-2, Months 1-9)

| Element | Detail |
|---------|--------|
| **Products** | Free OSS, Cloud Pay-as-you-go, Cloud Pro |
| **Funnel** | GitHub → pip install → try free → hit limit → upgrade to Cloud (self-serve via Stripe) |
| **Conversion triggers** | 10 free hours used up, need faster queues, want more concurrent clusters |
| **Payment** | Credit card via Stripe, automated metered billing |
| **Trial-to-paid** | No credit card required for free tier. Card required for any paid usage. |
| **Self-serve max** | Up to ~$2,000/month spend |
| **Support** | Discord community, email support tickets, knowledge base |

#### Sales-Assisted (Phase 2+, Months 6+)

| Element | Detail |
|---------|--------|
| **Products** | Cloud Pro (above $500/mo), Enterprise Self-Hosted, Enterprise Cloud |
| **Trigger** | Spend exceeds $500/mo, or prospect fills "Contact Sales" form |
| **Process** | Demo → POC (30-day) → Security review → Procurement → Onboarding |
| **Team** | 1 AE + 1 Solutions Engineer (hired Month 6) |
| **Target accounts** | Healthcare (HIPAA need), Financial services (SOC 2), Legal (data sovereignty), AI Labs (budget-constrained, multi-node) |

#### Sales-Led (Phase 3, Months 12+)

| Element | Detail |
|---------|--------|
| **Products** | Enterprise Self-Hosted, Enterprise Cloud |
| **Deal sizes** | $25K-100K ACV |
| **Team** | 2 AEs + 1 SE + 1 SDR |
| **Outbound** | SDR sequences targeting compliance-sensitive prospects |
| **Channel** | Direct + potential cloud marketplace (AWS/GCP/Azure) |
| **Sales cycle** | 45-90 days (typical for infrastructure software) |

### Why Hybrid Works for DistLLM

- **Self-serve captures the long tail**: Thousands of individual developers and indie builders never need a sales call. They pay $50-500/mo via credit card. Zero marginal sales cost.
- **Sales-led captures the whale**: Enterprise security and compliance requirements are complex and high-stakes. No amount of documentation replaces a conversation.
- **OSS is the top-of-funnel**: Every GitHub star, every pip install, every "Show HN" post is a free marketing impression. Enterprise deals often originate from an engineer who tried the OSS version first.
- **The upgrade path is natural**: Users self-serve until they hit a compliance or scale wall, at which point they're already sold on the product — they just need procurement and compliance paperwork.

---

## 8. Billing Needs

### Phase 1: MVP Billing (Months 1-6)

**Stack:** Stripe (core processing) + Stripe Billing (metered usage)

| Capability | Implementation |
|------------|---------------|
| Payment methods | Credit/debit card (Stripe elements) |
| Usage metering | DistLLM tracks GPU-hours per account, reports to Stripe via metered billing API |
| Pricing model | $0.15/GPU-hour, invoiced monthly in arrears |
| Free tier enforcement | Account-level tracking of free GPU-hour allowance, enforced at API level |
| Invoices | Stripe auto-generates monthly PDF invoices |
| Tax handling | Stripe Tax (automatic VAT/GST calculation) |
| Dunning | Stripe smart retries (3 attempts over 5 days) |
| Receipts | Stripe receipt emails |

**Key implementation detail:** Usage metering must be accurate within 5% and reconcileable. Every API call logs GPU-seconds consumed, aggregated hourly, reported to Stripe daily.

### Phase 2: Growth Billing (Months 6-18)

**Stack:** Stripe + Chargebee or Metronome for enterprise usage-based billing

| Capability | Implementation |
|------------|---------------|
| Prepaid credits | "Buy $100, get $120 in credit" — drives upfront cash collection |
| Commitment discounts | Monthly/quarterly/annual commitments prorated and invoiced |
| Multi-currency | USD, EUR, GBP — Stripe automatically presents local pricing |
| Invoicing | Enterprise invoices via Stripe Invoicing (PDF + portal) |
| Net terms | 15, 30, 45-day net terms for enterprise (requires credit check) |
| Self-serve plan changes | Users upgrade/downgrade/resume/cancel in billing portal |
| Coupon/promo system | Referral credits, launch promo codes, event discounts |
| Usage alerts | Email notification at 50%, 80%, 100% of forecast spend |
| Spend caps | Users set monthly GPU-hour budget; API throttles when exceeded |

### Phase 3: Enterprise Billing (Months 12+)

| Capability | Implementation |
|------------|---------------|
| Annual contracts | Invoiced annually with quarterly true-ups for overage |
| PO support | Purchase order number on invoices |
| Multi-entity billing | Parent company manages subsidiaries |
| Usage dashboards | Real-time spend dashboard for procurement visibility |
| Audit log export | Billing events exportable for internal audit |
| Consolidated invoicing | Multiple projects under one invoice |
| Subscription management | Enterprise portal with seat/node management |

### Billing Data Model (Core)

```python
# Simplified model — actual implementation is in src/distllm/billing/
class UsageEvent:
    account_id: str
    cluster_id: str
    gpu_type: str        # "RTX4090", "A100", etc.
    gpu_hours: Decimal    # e.g., 0.0042 (for a 15-second inference)
    timestamp: datetime
    tier: str             # "free", "payg", "pro"
    region: str           # "us-east", "eu-west"

class Invoice:
    account_id: str
    period_start: date
    period_end: date
    line_items: list[LineItem]
    total: Decimal
    credits_applied: Decimal
    balance_due: Decimal
    status: str           # "open", "paid", "overdue", "void"
```

---

## 9. Free Tier Limits for Virality

### Self-Hosted Free (OSS)

| Aspect | Policy |
|--------|--------|
| **Feature access** | Full open-core — no limits |
| **Duration** | Forever |
| **Scale** | Unlimited nodes (all on your hardware) |
| **Speed** | No artificial throttling |
| **Support** | Community (Discord, GitHub Issues) |
| **Updates** | Same release cadence as paid tiers |
| **Commercial use** | Permitted (Apache 2.0) |

**Why unlimited:** Self-hosted users pay with their own GPU hardware and electricity. DistLLM's cost is near-zero (GitHub hosting, CI). Every self-hosted deployment is a proof point, a potential upgrade path, and a source of word-of-mouth marketing.

### Cloud Free Tier

| Aspect | Policy |
|--------|--------|
| **GPU-hours** | 10 hours/month free |
| **Concurrent clusters** | 1 |
| **Max GPU count per cluster** | 2 GPUs |
| **Model size limit** | Up to 70B parameters |
| **Rate limits** | 10 RPM, 100K TPM |
| **Data retention** | Prompts/results not stored (in-memory only) |
| **Support** | Discord community |
| **Credit card required** | No |
| **Account expiry** | No — free tier never expires |
| **Referral bonus** | +5 hours per successful referral (both parties) |
| **Max referral bonus** | 50 hours/month (10 referrals) |

### Why These Limits Drive Virality

1. **10 GPU-hours is genuinely useful**
   - ~10 hours of sustained 70B inference
   - Enough to run dozens of experiments, build a prototype, or evaluate the product
   - Low commitment — users can try without believing they'll become paying customers

2. **No credit card barrier**
   - Zero friction to start
   - Industry data shows 3-5x conversion improvement vs requiring CC for trial
   - Users who reach the limit and need more are already sold on the product

3. **Referral program creates a viral loop**
   - Free tier user tells a friend → friend gets 5 hours → referrer gets 5 hours
   - Both become active users → both eventually hit the 10-hour cap → both upgrade
   - Network effect: the value of the product increases with more friends using it (shared GPU pools)

4. **The 10-hour cap is a natural conversion point**
   - Light users (1-2 hours/month) never hit the cap → they stay free forever (good marketing)
   - Heavy users hit the cap within days → they upgrade to Cloud Pay-as-you-go
   - The cap is tight enough to convert but generous enough to demonstrate value

5. **OSS is the ultimate viral funnel**
   - "Run 70B models for free on your own hardware" is a compelling message
   - Hacker News, Reddit, and Twitter love stories about avoiding cloud costs
   - Every OSS user becomes an implicit advocate: their setup is a demo
   - OSS users at universities, research labs, and startups become enterprise leads later

6. **The transition feels fair**
   - Users who self-host: never pay, never capped → they're the product (marketing, community)
   - Users who use the cloud: pay for what they use → the value proposition is clear
   - The free tier is limited but not frustrating — limits are about compute cost, not feature gating

### Viral KPI Targets

| Metric | Target (Year 1) |
|--------|-----------------|
| GitHub stars | 5,000+ |
| PyPI downloads | 50K+/month |
| Active self-hosted clusters | 500+ |
| Cloud free tier sign-ups | 2,000+ |
| Free → Paid conversion rate | 8-15% |
| Referral-driven sign-ups | 20% of total |
| Average referral chain length | 2.3 (each user refers 2+ others) |
| Viral coefficient (K) | >1.0 (each user brings >1 new user) |

### Ceiling Computations

| Parameter | Value |
|-----------|-------|
| Monthly free GPU-hours budget (at scale) | 20,000 hours (2,000 users × 10 hrs) |
| Cost of 20K GPU-hours at $0.07/avg | $1,400/month |
| As % of total Cloud revenue target ($15K MRR) | ~9% |
| **Free tier as marketing cost** | $1,400/month = $16,800/year |
| CAC equivalent at 2,000 sign-ups | $8.40 CAC (extremely efficient) |

The free tier's infrastructure cost is funded by the Cloud Pro tier margins. As long as free → paid conversion stays above 8%, the unit economics work.

---

## Summary: The Flywheel

```
                          ┌──────────────────────────────┐
                          │     OSS Self-Hosted          │
                          │  (Apache 2.0, fully free)    │
                          │   → GitHub stars, virality   │
                          └────────────┬─────────────────┘
                                       │
                          "I need more GPUs / less ops"
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │        DistLLM Cloud                │
                    │  Free tier (10 GPU-hr/mo)           │
                    │  → Try without commitment           │
                    │  → Upgrade when you hit the limit   │
                    └────────────┬────────────────────────┘
                                 │
                    "I need faster / more / priority"
                                 │
                                 ▼
                    ┌─────────────────────────────────────┐
                    │        Cloud Pro ($50/mo)           │
                    │  → Self-serve, credit card          │
                    │  → Priority, faster, 2x limits      │
                    └────────────┬────────────────────────┘
                                 │
                     "I need compliance / SSO / air-gap"
                                 │
                                 ▼
                    ┌─────────────────────────────────────┐
                    │    Enterprise ($1K/node/yr)         │
                    │  → Sales-led, contracts, compliance │
                    │  → SOC 2, HIPAA, SSO, audit         │
                    └─────────────────────────────────────┘
```

Each step is a natural upgrade. No feature gating on core capabilities. No artificial throttling. Users pay when they need more than the free tier provides — and the free tier provides a lot.

# Financial Projection Model
Assumptions and unit economics backing the monetization strategy in MARKETING.md.

---

## 1. Unit Economics

### Cloud Infrastructure Cost Model

| Item | Per GPU-Hour |
|------|-------------|
| GPU instance (reserved spot, bulk discount, e.g., RTX 4090 via Lambda Labs / Vast.ai) | $0.04-0.06 |
| Networking (bandwidth, load balancer) | $0.01-0.02 |
| Storage (model weights cache, KV cache disk) | $0.005-0.01 |
| Control plane overhead (API gateway, auth, metering) | $0.005 |
| **Total cost** | **$0.06-0.10** |
| **Blended revenue** | **$0.15** |
| **Gross margin** | **33-60%** (target: 50%+) |

### Unit Path to 70% Gross Margin

| Lever | Impact |
|-------|--------|
| Reserved/spot GPU instances | -30% GPU cost → $0.042/GPU-hr |
| Bulk discount at 10K+ GPU-hrs/month | -10% total cost |
| Commitment discounts shift mix toward upfront payment | Improves cash flow + reduces churn |
| Higher-margin Cloud Pro mix | Blended revenue rises to $0.17-0.18 |
| **Target steady-state** | **$0.17 revenue, $0.05 cost = 71% GM** |

---

## 2. Revenue Build-Up (Year 1-4)

All figures in thousands USD.

### Year 1 (Launch Year)

| Stream | Q1 | Q2 | Q3 | Q4 | Total |
|--------|----|----|----|----|-------|
| Cloud Pay-as-you-go | 0 | 2 | 8 | 20 | 30 |
| Cloud Pro | 0 | 0 | 3 | 10 | 13 |
| Enterprise (self-hosted) | 0 | 0 | 10 | 30 | 40 |
| Hub Marketplace | 0 | 0 | 0 | 2 | 2 |
| Ollama Plugin | 0 | 2 | 5 | 10 | 17 |
| **Total** | **0** | **4** | **26** | **72** | **102** |

**Year 1 ARR (exit): ~$250K** ($72K Q4 × 4, conservatively $288K projected, less churn/ramp)

### Year 2 (Growth)

| Stream | Revenue |
|--------|---------|
| Cloud Pay-as-you-go | 350 |
| Cloud Pro | 250 |
| Enterprise (self-hosted) | 500 |
| Hub Marketplace | 30 |
| Ollama Plugin | 120 |
| **Total** | **1,250** |

**Year 2 ARR (exit): ~$2.0M**

### Year 3 (Scale)

| Stream | Revenue |
|--------|---------|
| Cloud Pay-as-you-go | 1,500 |
| Cloud Pro | 1,200 |
| Enterprise (self-hosted) | 3,000 |
| Hub Marketplace | 200 |
| Ollama Plugin | 400 |
| **Total** | **6,300** |

**Year 3 ARR (exit): ~$10M**

### Year 4 (Market Leadership)

| Stream | Revenue |
|--------|---------|
| Cloud Pay-as-you-go | 5,000 |
| Cloud Pro | 4,000 |
| Enterprise (self-hosted) | 12,000 |
| Hub Marketplace | 1,000 |
| Ollama Plugin | 1,000 |
| **Total** | **23,000** |

**Year 4 ARR (exit): ~$50M** (original SOM target, revalidated)

---

## 3. Cohort Analysis (Cloud)

### By Month of First Paid Use

| Cohort | Users | Avg Monthly Spend | Month 1 Rev | Month 6 Rev | Month 12 Rev |
|--------|-------|-------------------|-------------|-------------|--------------|
| Free → Pay-as-you-go | 1,000 | $75 | $75K | $45K (60% retained) | $30K (40% retained) |
| Free → Cloud Pro | 200 | $150 | $30K | $24K (80% retained) | $20K (67% retained) |
| Enterprise | 20 | $4,000 | $80K | $72K (90% retained) | $64K (80% retained) |

### Retention Assumptions

| Tier | Monthly Net Retention | Annual Net Retention |
|------|----------------------|---------------------|
| Pay-as-you-go | 92% | 37% (logo) / 110%+ (dollar, due to usage growth) |
| Cloud Pro | 96% | 61% (logo) / 115%+ (dollar) |
| Enterprise | 97% | 69% (logo) / 120%+ (dollar, due to node expansion) |

---

## 4. Customer Acquisition Cost

### Paid Channels (Year 2+)

| Channel | Cost per Acquisition | Volume/Quarter | Notes |
|---------|---------------------|----------------|-------|
| GitHub OSS (organic) | $0 | 500+ | Primary funnel. Every star is free traffic. |
| Hacker News / Reddit | $0 | 50-100 | Organic content-driven. Blog posts, Show HNs. |
| YouTube demos | $0 (DIY) | 20-50 | Video production time, not cash. |
| AI newsletters | $500-2,000 | 10-30 | TLDR AI, TheSequence, etc. |
| Google Ads (branded) | $2-5 | 20-50 | Only after awareness is established. |
| Enterprise outbound | $5,000-10,000 | 5-10 | SDR salary + LinkedIn Sales Nav. |

### Blended CAC

| Phase | Blended CAC | Target Payback Period |
|-------|-------------|----------------------|
| Year 1 | ~$0 (all organic) | Immediate |
| Year 2 | ~$50 | <4 months |
| Year 3 | ~$100 | <6 months |

---

## 5. Cash Flow (Seed Stage)

### Assumptions

| Item | Monthly Cost |
|------|-------------|
| Engineering (2 FTEs) | $25K |
| GPU test hardware (amortized) | $2K |
| Cloud infra (CI, staging, prod) | $5K |
| Legal & compliance (amortized) | $3K |
| Marketing (content, tools) | $2K |
| **Total burn** | **$37K** |
| **Monthly revenue (Y1 avg)** | **~$8.5K** |
| **Net burn** | **~$28.5K** |
| **Runway on $500K seed** | **~17.5 months** |
| **Runway to break-even** | **~14 months (Y1 Y2 boundary)** |

---

## 6. Key Metrics Target

| Metric | Year 1 Target | Year 2 Target | Year 3 Target |
|--------|---------------|---------------|---------------|
| ARR | $250K | $2M | $10M |
| MRR | $20K | $170K | $830K |
| Gross margin | 30% | 50% | 65% |
| Net revenue retention | 105% | 115% | 120% |
| Free users | 2,000 | 10,000 | 50,000 |
| Paid accounts | 150 | 1,000 | 5,000 |
| Enterprise accounts | 5 | 30 | 150 |
| CAC payback | Immediate | 4 months | 6 months |
| LTV / CAC | N/A (organic) | 12x | 15x |
| NPS | 40+ | 50+ | 60+ |
| Employees | 2 | 6 | 15 |

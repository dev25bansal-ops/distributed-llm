# Export Controls

Compliance with AI export regulations for DistLLM.

---

## Overview

DistLLM is an open-source distributed LLM inference framework. Users are responsible for ensuring their use of DistLLM complies with applicable export control laws and regulations.

---

## Applicable Regulations

### United States

- **Export Administration Regulations (EAR)**: Administered by the Bureau of Industry and Security (BIS)
- **Entity List**: Users must ensure they are not on the BIS Entity List
- **End-Use Restrictions**: AI technology may be restricted for certain end uses (e.g., weapons of mass destruction)

### European Union

- **EU Dual-Use Regulation**: Regulation (EU) 2021/821
- **AI Act**: Regulation (EU) 2024/1689 — high-risk AI systems may require conformity assessment

### International

- **Wassenaar Arrangement**: Dual-use technologies including AI/ML
- **Nuclear Suppliers Group**: AI for nuclear applications

---

## User Responsibilities

### 1. Know Your Customer (KYC)

Users deploying DistLLM as a managed service must:
- Verify customer identity
- Screen against sanctioned entity lists
- Document end-use declarations

### 2. End-Use Monitoring

Users must ensure DistLLM is not used for:
- Development of weapons of mass destruction
- Military end-use in restricted countries
- Surveillance systems violating human rights
- Other prohibited end-uses under applicable law

### 3. Technical Data

When sharing DistLLM technical data (documentation, source code, configurations):
- Comply with applicable export controls
- Do not share with sanctioned entities
- Document any deemed exports

---

## Self-Hosted Deployment

Self-hosted DistLLM deployments are subject to:
- Local export control laws of the deployment country
- End-use restrictions of the model being served
- Data residency requirements

---

## Model-Specific Considerations

Different models may have their own export restrictions:
- Check the model's license on HuggingFace
- Some models (e.g., certain Chinese models) may have specific restrictions
- Users are responsible for model-specific compliance

---

## Compliance Checklist

- [ ] Verify not on any sanctions/entity list
- [ ] Document intended end-use
- [ ] Check model-specific export restrictions
- [ ] Implement access controls (RBAC)
- [ ] Enable audit logging
- [ ] Configure data residency as required
- [ ] Review local export control laws

---

## Disclaimer

This document is for informational purposes only and does not constitute legal advice. Users should consult with legal counsel to ensure compliance with applicable export control laws.

---

## Contact

For export control inquiries: compliance@distllm.dev

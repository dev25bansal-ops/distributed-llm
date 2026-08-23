"""Compliance / certification evidence helpers for DistLLM.

This sub-package builds on N4's point-in-time SOC 2 / ISO 27001 evidence
collector (``distllm.core.compliance_evidence``) and assembles a broader,
auditor-facing **evidence pack** that also folds in the doc-derived control
mappings for GDPR, Export Controls, and HIPAA (mapped from the security
posture where no dedicated source doc exists).

See ``evidence_pack.py`` for the public API.
"""

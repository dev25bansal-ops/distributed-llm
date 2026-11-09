"""Model accuracy verification — compare distributed vs single-node inference.

Provides a framework for verifying that distributed pipeline-parallel
inference produces outputs that match a single-node reference within
configurable tolerances. Detects numerical drift from:

  - INT8/Fp8 activation quantization between pipeline stages
  - Layer partitioning across devices
  - gRPC serialization/deserialization
  - KV cache delta transfer

Usage:
    from distllm.verification.runner import AccuracyVerifier

    verifier = AccuracyVerifier(
        model_name="HuggingFaceTB/SmolLM-135M",
        num_nodes=2,
    )
    report = await verifier.verify(prompts=["The capital of France is"])
    print(report.summary())
"""

from .comparator import (
    OutputComparison,
    compare_hidden_states,
    compare_logits,
    compare_text,
    compare_tokens,
)
from .hash_registry import GenerationOutput, OutputHashRegistry, compute_output_hash
from .report import VerificationReport, generate_report
from .runner import AccuracyVerifier, run_verification_cli, verify_accuracy

__all__ = [
    "OutputComparison",
    "compare_logits",
    "compare_hidden_states",
    "compare_tokens",
    "compare_text",
    "compute_output_hash",
    "GenerationOutput",
    "OutputHashRegistry",
    "VerificationReport",
    "generate_report",
    "print_report",
    "AccuracyVerifier",
    "verify_accuracy",
    "run_verification_cli",
]

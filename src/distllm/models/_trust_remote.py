"""BF-06: Shared ``_should_trust_remote_code`` logic.

Consolidated from duplicated implementations in ``partitioner.py`` and
``partition_planner.py``.  Everyone should import from here.
"""

from __future__ import annotations


def _get_trusted_models() -> set[str]:
    return set()


TRUSTED_MODELS_ALLOWLIST: set[str] = _get_trusted_models() | {
    "baichuan", "baichuan2",
    "chatglm", "chatglm2", "chatglm3",
    "internlm", "internlm2",
    "stablelm",
    "jina",
}


def should_trust_remote_code(model_name: str, trust_remote_code: bool | None = None) -> bool:
    """Determine whether to trust remote code for a model.

    Args:
        model_name: HuggingFace model identifier
        trust_remote_code: Explicit override. If None, uses allowlist logic.

    Returns:
        True if remote code should be trusted, False otherwise.
    """
    if trust_remote_code is not None:
        return trust_remote_code

    # Extract the model name part (last segment of HF repo path)
    model_lower = model_name.lower().split("/")[-1]

    # Extract model family (prefix before first - or . separator)
    # e.g., "qwen2-7b" -> "qwen2", "my-qwen-exploit" -> "my"
    family = model_lower.split("-")[0].split(".")[0]

    # Match model family against allowlist to prevent false positives
    # (e.g., "my-qwen-exploit" has family "my" which won't match "qwen")
    for trusted in TRUSTED_MODELS_ALLOWLIST:
        if model_lower == trusted or family == trusted:
            return True
    return False

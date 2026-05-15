"""Built-in chat template functions for common model families."""

from typing import List, Dict


def chatml_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """ChatML format: <|im_start|>role\ncontent<|im_end|>"""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def llama2_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Llama-2 format: <s>[INST] <<SYS>>...<</SYS>> ... [/INST]"""
    parts = ["<s>"]
    system_content = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_content = content
        elif role == "user":
            if system_content:
                parts.append(f"[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{content} [/INST]")
                system_content = ""
            else:
                parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            parts.append(f" {content} </s>")
    if add_generation_prompt and not parts[-1].endswith("</s>"):
        pass  # Llama-2 doesn't need a special generation prompt after [/INST]
    return "".join(parts)


def llama3_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Llama-3 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>"""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def mistral_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Mistral format: <s>[INST] ... [/INST]"""
    parts = ["<s>"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            parts.append(f" {content}")
        elif role == "system":
            parts.append(f"[INST] {content} [/INST]")
    if add_generation_prompt:
        pass  # Mistral doesn't need special generation prompt
    return "".join(parts)


def zephyr_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Zephyr format: <|system|>\n...<|user|>\n...<|assistant|>\n..."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts) + "\n"


def alpaca_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Alpaca format: ### Instruction:\n...\n\n### Response:\n..."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "system"):
            parts.append(f"### Instruction:\n{content}")
        elif role == "assistant":
            parts.append(f"### Response:\n{content}")
    if add_generation_prompt:
        parts.append("### Response:\n")
    return "\n\n".join(parts)


# Registry of built-in templates
BUILTIN_TEMPLATES = {
    "chatml": chatml_template,
    "llama-2": llama2_template,
    "llama2": llama2_template,
    "llama-3": llama3_template,
    "llama3": llama3_template,
    "mistral": mistral_template,
    "zephyr": zephyr_template,
    "alpaca": alpaca_template,
}

# Model name patterns for auto-detection
MODEL_PATTERNS = {
    "llama-3": ["llama-3", "llama3"],
    "llama-2": ["llama-2", "llama2"],
    "mistral": ["mistral"],
    "zephyr": ["zephyr"],
    "alpaca": ["alpaca"],
    "chatml": ["chatml", "qwen", "yi"],  # Qwen and Yi use ChatML
}


def auto_detect_template(model_name: str) -> str:
    """Auto-detect template name from model name.

    Returns:
        Template name or "auto" if no match found.
    """
    model_lower = model_name.lower()
    for template_name, patterns in MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern in model_lower:
                return template_name
    return "auto"

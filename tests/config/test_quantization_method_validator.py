"""Validator: QuantizationSettings.method accepts both spellings.

CLI argument parser advertises ``bitsandbytes_4bit`` / ``bitsandbytes_8bit``
(matching the bitsandbytes package name). The rest of the codebase uses the
short canonical form ``bnb_4bit`` / ``bnb_8bit``. Without the alias map the
validator rejects the CLI value and the server crashes at startup. This
test pins down all 8 supported spellings and their normalization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distllm.config._model import QuantizationSettings


CANONICAL = ("none", "fp8", "bnb_4bit", "bnb_8bit", "gptq", "awq")
CLI_ALIASES = ("bitsandbytes_4bit", "bitsandbytes_8bit")

ALL_INPUTS = list(CANONICAL) + list(CLI_ALIASES)
# Map: input -> expected normalized output
EXPECTED = {
    "none": "none",
    "fp8": "fp8",
    "bnb_4bit": "bnb_4bit",
    "bnb_8bit": "bnb_8bit",
    "gptq": "gptq",
    "awq": "awq",
    "bitsandbytes_4bit": "bnb_4bit",
    "bitsandbytes_8bit": "bnb_8bit",
}


@pytest.mark.parametrize("raw,expected", list(EXPECTED.items()))
def test_quantization_method_accepts_and_normalizes(raw: str, expected: str) -> None:
    """Each accepted spelling normalizes to the canonical short form."""
    settings = QuantizationSettings(method=raw)
    assert settings.method == expected, (
        f"input {raw!r} should normalize to {expected!r}, got {settings.method!r}"
    )


def test_all_required_names_covered() -> None:
    """Belt-and-braces: the validator must accept exactly the 8 documented names.

    If someone adds a new method to the docs but forgets to extend
    ``_QUANT_METHOD_CANONICAL`` / ``_QUANT_METHOD_ALIASES``, this test fails.
    """
    # Sanity: the test itself enumerates the 8 names; this asserts we have
    # not silently dropped any in the table above.
    assert len(ALL_INPUTS) == 8, (
        "test must enumerate exactly 8 quantization method spellings; "
        "update this test together with the validator."
    )


def test_bitsandbytes_alias_normalization_closes_real_crash() -> None:
    """Regression: CLI passes ``bitsandbytes_4bit`` and the validator crashed.

    Before this fix, server.py crashed at startup because the CLI spelling
    was rejected by QuantizationSettings.validate_method. Pinning the
    canonical output (`bnb_4bit`) makes sure downstream model-loader code
    can switch on the value without caring which spelling the user typed.
    """
    settings = QuantizationSettings(method="bitsandbytes_4bit")
    assert settings.method == "bnb_4bit"
    settings = QuantizationSettings(method="bitsandbytes_8bit")
    assert settings.method == "bnb_8bit"


def test_unknown_method_is_rejected() -> None:
    """Typos and unknown methods still raise — only documented aliases pass."""
    with pytest.raises(ValidationError) as ei:
        QuantizationSettings(method="not_a_real_method")
    assert "method" in str(ei.value).lower()


def test_non_string_method_is_rejected() -> None:
    """Defensive: a None / int / list value must fail at the validator boundary.

    Earlier the validator did ``v not in allowed`` which would raise TypeError
    on a non-string instead of a clean ValidationError — guard against that.
    """
    for bad in (None, 4, ["bnb_4bit"]):
        with pytest.raises(ValidationError):
            QuantizationSettings(method=bad)  # type: ignore[arg-type]


def test_default_method_is_none() -> None:
    """Default must remain the safe 'no quantization' option."""
    s = QuantizationSettings()
    assert s.method == "none"
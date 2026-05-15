"""E2E tests for model loading (requires actual model download)."""

import pytest


@pytest.mark.e2e
@pytest.mark.slow
class TestModelLoadingE2E:
    """These tests download real models and require network access.

    Skipped by default with `-m "not slow"`.
    """

    def test_tinystories_model_generates_output(self):
        """Load TinyStories-1M and verify it generates non-empty output.

        This test downloads a ~4MB model from HuggingFace.
        """
        # Skip if no network or huggingface-hub not installed
        pytest.importorskip("transformers")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "roneneldan/TinyStories-1M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Simple generation
        inputs = tokenizer("Once upon a time", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert len(text) > 0
        assert "Once upon a time" in text

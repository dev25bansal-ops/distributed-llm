"""Synthetic data generation for training and evaluation.

Provides ``SynthDataGenerator`` for generating, augmenting, and
exporting synthetic datasets via the DistLLM API.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from distllm_sdk.errors import InvalidRequestError

_logger = logging.getLogger(__name__)


_PATH_TRAVERSAL_RE = re.compile(r"(\.\.(\\|/)|~/|/etc/|/var/|/proc/|/sys/|/dev/)", re.IGNORECASE)


def _validate_export_path(filepath: str) -> Path:
    """Validate an export file path (no traversal, writable parent directory)."""
    path = Path(filepath).resolve()
    if _PATH_TRAVERSAL_RE.search(str(path)):
        raise InvalidRequestError(f"Path contains invalid patterns: {filepath}")
    parent = path.parent
    if not parent.exists():
        raise InvalidRequestError(f"Parent directory does not exist: {parent}")
    if not os.access(str(parent), os.W_OK):
        raise InvalidRequestError(f"Parent directory is not writable: {parent}")
    return path


# -- Built-in prompt templates -------------------------------------------------

_BUILTIN_TEMPLATES: dict[str, str] = {
    "qa": (
        "You are a synthetic data generator. Generate {num_examples} unique "
        "question-answer pairs on the topic(s): {topics}.\n\n"
        "Each pair must:\n"
        "- Be self-contained and realistic\n"
        "- Vary in difficulty from easy to expert\n"
        "- Cover different subtopics within each topic\n\n"
        "Return a JSON list of objects with keys \"question\" and \"answer\".\n"
        "Do not wrap in markdown code fences; output raw JSON only."
    ),
    "summarization": (
        "You are a synthetic data generator. Generate {num_examples} distinct "
        "text-summary pairs on the topic(s): {topics}.\n\n"
        "Each pair must:\n"
        "- Have a source text of 3-5 sentences\n"
        "- Have a concise 1-2 sentence summary\n"
        "- Cover different angles or subtopics\n\n"
        "Return a JSON list of objects with keys \"text\" and \"summary\".\n"
        "Do not wrap in markdown code fences; output raw JSON only."
    ),
    "classification": (
        "You are a synthetic data generator. Generate {num_examples} "
        "text-label pairs for classification on the topic(s): {topics}.\n\n"
        "Each pair must:\n"
        "- Have a realistic input text (1-3 sentences)\n"
        "- Have exactly one label from a plausible set of categories\n"
        "- Cover edge cases and unambiguous examples\n\n"
        "Return a JSON list of objects with keys \"text\" and \"label\".\n"
        "Do not wrap in markdown code fences; output raw JSON only."
    ),
    "conversation": (
        "You are a synthetic data generator. Generate {num_examples} distinct "
        "multi-turn dialogues on the topic(s): {topics}.\n\n"
        "Each dialogue must:\n"
        "- Have 3-6 alternating turns between user and assistant\n"
        "- Be coherent and conversational\n"
        "- Cover different subtopics and interaction patterns\n\n"
        "Return a JSON list of objects with key \"messages\" whose value is a "
        "list of {\"role\": \"user\"|\"assistant\", \"content\": \"...\"} dicts.\n"
        "Do not wrap in markdown code fences; output raw JSON only."
    ),
}

_DEFAULT_TEMPLATE = _BUILTIN_TEMPLATES["qa"]


@dataclass(frozen=True)
class SynthDataConfig:
    """Configuration for synthetic data generation.

    Attributes:
        model: Model name to use for generation.
        num_examples: Number of examples to generate per call.
        max_tokens: Maximum tokens for each generated response.
        temperature: Sampling temperature (lower = more deterministic).
        topics: List of topics or domains for the generated data.
    """

    model: str = "distributed-llm"
    num_examples: int = 10
    max_tokens: int = 4096
    temperature: float = 0.8
    topics: list[str] = field(default_factory=lambda: ["general"])


class SynthDataGenerator:
    """Generate, augment, and export synthetic training/evaluation data.

    Uses ``httpx.AsyncClient`` to send chat-completion requests to a
    DistLLM API endpoint.  All generation methods return plain ``list[dict]``
    that can be post-processed or exported via ``to_jsonl`` / ``to_csv``.

    Typical usage::

        config = SynthDataConfig(
            model="distributed-llm",
            num_examples=20,
            topics=["python", "machine learning"],
        )
        gen = SynthDataGenerator(
            base_url="http://localhost:8000",
            api_key="sk-...",
            config=config,
        )
        data = await gen.generate(prompt_template="qa")
        gen.to_jsonl(data, "output.jsonl")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        config: SynthDataConfig | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the generator.

        Args:
            base_url: DistLLM API base URL.
            api_key: Optional API key (Bearer token).
            config: Generation configuration.
            timeout: HTTP request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._config = config or SynthDataConfig()
        self._timeout = timeout

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> SynthDataGenerator:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # -- Properties -----------------------------------------------------------

    @property
    def config(self) -> SynthDataConfig:
        return self._config

    @config.setter
    def config(self, value: SynthDataConfig) -> None:
        self._config = value

    # -- Internal helpers -----------------------------------------------------

    def _format_prompt(self, template: str) -> str:
        """Format a prompt template with the current config values."""
        topics_str = ", ".join(self._config.topics)
        return template.format(
            num_examples=self._config.num_examples,
            topics=topics_str,
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

    def _build_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Build the message list for a chat completion request."""
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Generate {self._config.num_examples} examples now. "
                    "Return only valid JSON as specified."
                ),
            },
        ]

    async def _request_completion(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Send a chat completion request and return the parsed JSON body."""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    def _extract_content(self, response_data: dict[str, Any]) -> str:
        """Extract the text content from a chat completion response."""
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices in API response")
        message = choices[0].get("message", {})
        content: str = message.get("content", "")
        return content

    @staticmethod
    def _parse_json(content: str) -> list[dict[str, Any]]:
        """Parse a JSON list from the LLM response, stripping code fences."""
        cleaned = content.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            # Find the first newline after the opening fence
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1 :]
            # Remove trailing fence
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
            elif "```" in cleaned:
                cleaned = cleaned[: cleaned.rfind("```")].rstrip()
        # Try to extract a JSON array if there is extra text
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")
        if array_start != -1 and array_end > array_start:
            cleaned = cleaned[array_start : array_end + 1]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse generated content as JSON: {exc}\n"
                f"Raw content preview: {content[:300]}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(
                f"Expected a JSON list, got {type(parsed).__name__}"
            )
        return parsed

    # -- Core generation ------------------------------------------------------

    async def generate(
        self,
        prompt_template: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate synthetic data using a prompt template.

        If *prompt_template* is ``None``, the built-in ``"qa"`` template is
        used.  Otherwise *prompt_template* can be one of the built-in template
        names (``"qa"``, ``"summarization"``, ``"classification"``,
        ``"conversation"``) or a custom format-string that accepts
        ``{num_examples}`` and ``{topics}`` placeholders.

        Args:
            prompt_template: Template name, custom format string, or ``None``.

        Returns:
            A list of dicts, each representing one generated example.

        Raises:
            ValueError: If the API response cannot be parsed as a JSON list.
            httpx.HTTPStatusError: On API error responses.
        """
        if prompt_template is None:
            template_str = _BUILTIN_TEMPLATES["qa"]
        elif prompt_template in _BUILTIN_TEMPLATES:
            template_str = _BUILTIN_TEMPLATES[prompt_template]
        else:
            template_str = prompt_template

        system_prompt = self._format_prompt(template_str)
        messages = self._build_messages(system_prompt)
        response_data = await self._request_completion(messages)
        content = self._extract_content(response_data)
        parsed = self._parse_json(content)
        _logger.info(
            "Generated %d examples using model=%s template=%s",
            len(parsed),
            self._config.model,
            prompt_template or "qa",
        )
        return parsed

    async def generate_from_schema(
        self,
        schema: dict[str, Any],
        num_examples: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate synthetic data conforming to a JSON schema.

        Args:
            schema: A JSON Schema dict describing the desired output structure.
            num_examples: Number of examples to generate.

        Returns:
            A list of dicts where each dict conforms to *schema*.
        """
        schema_str = json.dumps(schema, indent=2)
        system_prompt = (
            f"You are a synthetic data generator. Generate {num_examples} "
            f"distinct valid JSON objects that conform to this JSON Schema "
            f"on the topic(s): {', '.join(self._config.topics)}.\n\n"
            f"Schema:\n{schema_str}\n\n"
            "Each object must strictly match the schema types and constraints. "
            "Vary the field values across examples for diversity.\n\n"
            "Return a JSON list of the generated objects. "
            "Do not wrap in markdown code fences; output raw JSON only."
        )
        # Temporarily override num_examples for this call
        original_num = self._config.num_examples
        try:
            object.__setattr__(self._config, "num_examples", num_examples)
            messages = self._build_messages(system_prompt)
            response_data = await self._request_completion(messages)
            content = self._extract_content(response_data)
            parsed = self._parse_json(content)
            _logger.info(
                "Generated %d schema-conforming examples using model=%s",
                len(parsed),
                self._config.model,
            )
            return parsed
        finally:
            object.__setattr__(self._config, "num_examples", original_num)

    async def augment(
        self,
        examples: list[dict[str, Any]],
        variations: int = 3,
    ) -> list[dict[str, Any]]:
        """Create variations of existing examples.

        Each input example is rephrased, reworded, or augmented with
        controlled noise to produce *variations* new examples.

        Args:
            examples: The original examples to augment.
            variations: Number of variations to produce per example.

        Returns:
            A new list of dicts containing the original examples plus their
            variations.
        """
        schema_str = json.dumps(examples[:3], indent=2)  # show shape
        total_target = len(examples) * (variations + 1)
        system_prompt = (
            f"You are a synthetic data augmenter. You are given {len(examples)} "
            f"example record(s). Create {variations} new variation(s) for EACH "
            f"original record by rephrasing, rewording, or adding controlled "
            f"noise while preserving the structure and meaning.\n\n"
            f"First {min(3, len(examples))} original record(s) for reference:\n"
            f"{schema_str}\n\n"
            f"Return exactly {total_target} JSON objects (originals first, "
            f"then variations) as a single JSON list. "
            "Do not wrap in markdown code fences; output raw JSON only."
        )
        system_prompt = system_prompt.replace(
            "{variations}", str(variations)
        ).replace("{total_target}", str(total_target))

        original_num = self._config.num_examples
        try:
            # Use a large-enough number for the meta-request
            object.__setattr__(self._config, "num_examples", total_target)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Original examples:\n{json.dumps(examples, indent=2)}\n\n"
                        f"Generate {variations} variation(s) for each example, "
                        f"returning all {total_target} objects as a JSON list."
                    ),
                },
            ]
            response_data = await self._request_completion(messages)
            content = self._extract_content(response_data)
            parsed = self._parse_json(content)
            _logger.info(
                "Augmented %d examples into %d total (%.1fx)",
                len(examples),
                len(parsed),
                len(parsed) / max(len(examples), 1),
            )
            return parsed
        finally:
            object.__setattr__(self._config, "num_examples", original_num)

    # -- Export helpers -------------------------------------------------------

    def to_jsonl(
        self,
        data: list[dict[str, Any]],
        filepath: str,
    ) -> None:
        """Export generated data to a JSON Lines file.

        Args:
            data: List of dicts to export.
            filepath: Output file path.

        Raises:
            InvalidRequestError: If the path is unsafe or unwritable.
        """
        path = _validate_export_path(filepath)
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _logger.info("Wrote %d records to %s", len(data), path)

    def to_csv(
        self,
        data: list[dict[str, Any]],
        filepath: str,
        fieldnames: list[str] | None = None,
    ) -> None:
        """Export generated data to a CSV file.

        Args:
            data: List of dicts to export.
            filepath: Output file path.
            fieldnames: Column order.  If ``None``, inferred from the first
                record's keys.

        Raises:
            InvalidRequestError: If the path is unsafe or unwritable.
        """
        path = _validate_export_path(filepath)
        if not data:
            # Create an empty file
            path.write_text("", encoding="utf-8")
            _logger.info("Wrote empty CSV to %s", path)
            return

        columns = fieldnames or list(data[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for record in data:
                writer.writerow(record)
        _logger.info("Wrote %d records to %s", len(data), path)

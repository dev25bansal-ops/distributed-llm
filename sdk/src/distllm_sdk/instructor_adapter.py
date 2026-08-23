"""Instructor adapter for DistLLM — structured extraction.

Allows using DistLLM with the ``instructor`` library for type-safe,
structured data extraction from LLM outputs.

Usage::

    from pydantic import BaseModel
    from distllm_sdk.instructor_adapter import DistLLMInstructor

    class UserDetail(BaseModel):
        name: str
        age: int

    client = DistLLMInstructor(base_url="http://localhost:8000")
    user, raw = client.extract(
        "John is 25 years old",
        response_model=UserDetail,
    )
    print(user.name, user.age)  # John 25
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("distllm_sdk")


class DistLLMInstructor:
    """Structured extraction client backed by a DistLLM cluster.

    Works with or without the ``instructor`` package.  When ``instructor``
    is installed, uses ``instructor.from_openai()`` with the OpenAI-compat
    client.  Otherwise falls back to JSON-mode extraction.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 120.0,
        mode: str = "json",
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._mode = mode
        self._instructor_available = False

        import httpx
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(timeout),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _openai_client(self) -> Any:
        """Return a minimal OpenAI-compatible client for instructor."""
        from types import SimpleNamespace

        class _InnerCompletions:
            def __init__(self, parent: DistLLMInstructor):
                self._parent = parent

            def create(self, **kwargs: Any) -> Any:
                resp = self._parent._client.post(
                    "/v1/chat/completions",
                    json=kwargs,
                    headers=self._parent._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                return SimpleNamespace(
                    id=data.get("id", ""),
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(
                            role=msg.get("role", "assistant"),
                            content=msg.get("content", ""),
                            tool_calls=None,
                        ),
                        finish_reason=choice.get("finish_reason", "stop"),
                    )],
                    usage=SimpleNamespace(**data.get("usage", {})),
                )

        class _InnerChat:
            def __init__(self, parent: DistLLMInstructor):
                self.completions = _InnerCompletions(parent)

        return SimpleNamespace(chat=_InnerChat(self))

    def extract(
        self,
        text: str,
        response_model: type,
        system_prompt: str | None = None,
        model: str = "distributed-llm",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> tuple[Any, dict]:
        """Extract structured data from *text*.

        When ``instructor`` is installed, uses the instructor patch for
        reliable type-safe extraction.  Otherwise uses JSON-mode with retry.

        Args:
            text: Input text to extract from.
            response_model: A Pydantic BaseModel class (or other type with
                ``model_json_schema()`` or ``__dataclass_fields__``).
            system_prompt: Optional system prompt override.
            model: Model to use.
            temperature: Sampling temperature (0 = deterministic).
            max_retries: Number of retries on parse failure.

        Returns:
            Tuple of (parsed_model, raw_response_dict).
        """
        # Try instructor path
        try:
            import instructor
            from pydantic import BaseModel

            if issubclass(response_model, BaseModel) and self._instructor_available:
                client = instructor.from_openai(self._openai_client(), mode=self._mode)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": text})

                result = client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    messages=messages,
                    temperature=temperature,
                )
                return result, result.model_dump() if hasattr(result, "model_dump") else {}

        except ImportError:
            self._instructor_available = False
        except Exception as e:
            logger.warning("Instructor extraction failed, falling back: %s", e)

        # Fallback: JSON-mode extraction
        return self._extract_json(text, response_model, model, system_prompt, temperature, max_retries)

    def _extract_json(
        self,
        text: str,
        response_model: type,
        model: str,
        system_prompt: str | None,
        temperature: float,
        max_retries: int,
    ) -> tuple[Any, dict]:
        """Fallback extraction using JSON response_format."""
        import json as _json

        # Get schema
        schema = self._get_schema(response_model)
        schema_str = _json.dumps(schema, indent=2) if schema else "{}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": (
                f"Extract the requested information from the following text. "
                f"Respond ONLY with a valid JSON object matching this schema:\n"
                f"```json\n{schema_str}\n```\n\n"
                f"Text:\n{text}"
            ),
        })

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 1024,
                    "response_format": {"type": "json_object"},
                }
                resp = self._client.post("/v1/chat/completions", json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Strip markdown code fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                    content = content.rsplit("\n```", 1)[0]

                parsed = _json.loads(content)
                return self._to_model(parsed, response_model), parsed

            except Exception as e:
                last_error = e
                logger.warning("JSON extraction attempt %d failed: %s", attempt + 1, e)
                continue

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _get_schema(model_type: type) -> dict | None:
        """Get JSON schema from a type."""
        try:
            if hasattr(model_type, "model_json_schema"):
                return model_type.model_json_schema()
            if hasattr(model_type, "__dataclass_fields__"):
                from pydantic.dataclasses import dataclass
                # Simple field listing
                fields = {}
                for fname, ffield in model_type.__dataclass_fields__.items():
                    ftype = str(ffield.type) if hasattr(ffield.type, "__name__") else str(ffield.type)
                    fields[fname] = {"type": ftype}
                return {"type": "object", "properties": fields, "required": list(fields.keys())}
        except Exception:
            pass
        return None

    @staticmethod
    def _to_model(data: dict, model_type: type) -> Any:
        """Convert a dict to a model instance."""
        try:
            if hasattr(model_type, "model_validate"):
                return model_type.model_validate(data)
            if hasattr(model_type, "parse_obj"):
                return model_type.parse_obj(data)
            if hasattr(model_type, "__dataclass_fields__"):
                return model_type(**data)
            return data
        except Exception as e:
            raise ValueError(f"Failed to construct {model_type.__name__}: {e}") from e

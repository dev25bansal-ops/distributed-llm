"""Model search across public catalogs.

Onboarding needs to answer "what can this machine run?", which means querying
real catalogs rather than shipping a list. Two sources are supported today:

* **Hugging Face** -- the de-facto catalog for open weights, with a public
  search API and, importantly, a `safetensors` block carrying the exact
  parameter count, which is what makes honest memory math possible.
* **Ollama library** -- a curated set of GGUF builds, convenient for users who
  already run Ollama locally.

Both are queried the same way and normalised into :class:`ModelCandidate`, so
the recommender does not care where a model came from.

Nothing here downloads a model; it only discovers what exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

HF_SEARCH_URL = "https://huggingface.co/api/models"
OLLAMA_LIBRARY_URL = "https://ollama.com/library"

# Tags/suffixes that mark a repository as not a text-generation model we can
# serve: embeddings, classifiers, vision encoders, and adapter-only repos.
_EXCLUDE_PIPELINE = {
    "text-classification",
    "token-classification",
    "feature-extraction",
    "image-classification",
    "object-detection",
    "automatic-speech-recognition",
    "text-to-speech",
    "text-to-image",
    "fill-mask",
    "sentence-similarity",
}

# 1 billion is close enough for ordering; the exact count comes from the API
# when available and this only fills gaps.
_PARAM_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
# Mixture-of-experts names look like "35B-A3B": 35B total parameters, 3B active
# per token. Sizing must use the total (that is what has to be resident); the
# active count only affects speed.
_MOE_NAME = re.compile(r"(\d+(?:\.\d+)?)[bB][-_]a(\d+(?:\.\d+)?)[bB]", re.IGNORECASE)


@dataclass
class ModelCandidate:
    """A model found in a catalog, normalised."""

    model_id: str
    source: str
    params: int = 0
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    dtype: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.model_id.rsplit("/", 1)[-1]

    @property
    def size_label(self) -> str:
        if self.params:
            return f"{self.params / 1e9:.1f}B"
        return "?"


def _params_from_name(name: str) -> int:
    """Best-effort parameter count from a repo name like ``Llama-3.1-8B``.

    For mixture-of-experts names (``35B-A3B``) this returns the *total*, which
    is what must fit in memory; the active count only affects how fast it runs.
    """
    moe = _MOE_NAME.search(name)
    if moe:
        try:
            return int(float(moe.group(1)) * 1e9)
        except ValueError:
            pass
    matches = _PARAM_SUFFIX.findall(name)
    if not matches:
        return 0
    try:
        return int(float(matches[-1]) * 1e9)
    except ValueError:
        return 0


def _moe_note(name: str) -> str:
    """Describe a mixture-of-experts model, or return an empty string."""
    moe = _MOE_NAME.search(name)
    if not moe:
        return ""
    return f"MoE: {moe.group(1)}B total, {moe.group(2)}B active per token"


def search_huggingface(
    query: str = "",
    *,
    limit: int = 30,
    timeout: float = 20.0,
    text_generation_only: bool = True,
) -> list[ModelCandidate]:
    """Search the Hugging Face model hub.

    Results are ordered by downloads. Parameter counts come from the
    ``safetensors`` metadata when the API provides it; that is the number the
    recommender needs for its memory estimate, so repositories that do not
    expose it are still returned but flagged.
    """
    params: dict[str, object] = {
        "limit": str(min(max(limit, 1), 100)),
        "sort": "downloads",
        "direction": "-1",
        "full": "true",
    }
    if query.strip():
        params["search"] = query.strip()
    if text_generation_only:
        params["filter"] = "text-generation"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(HF_SEARCH_URL, params=params)
            response.raise_for_status()
            raw = response.json()
    except (httpx.HTTPError, ValueError):
        return []

    out: list[ModelCandidate] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id") or ""
        if not model_id:
            continue

        pipeline = entry.get("pipeline_tag") or ""
        if text_generation_only and pipeline in _EXCLUDE_PIPELINE:
            continue

        safetensors = entry.get("safetensors") or {}
        total = safetensors.get("total") if isinstance(safetensors, dict) else None
        n_params = int(total) if isinstance(total, int) and total > 0 else 0

        dtypes = safetensors.get("parameters") if isinstance(safetensors, dict) else None
        dtype = ""
        if isinstance(dtypes, dict) and dtypes:
            dtype = str(next(iter(dtypes))).lower()

        notes: list[str] = []
        moe = _moe_note(model_id)
        if moe:
            notes.append(moe)
        if not n_params:
            n_params = _params_from_name(model_id)
            if n_params:
                notes.append("size estimated from the repository name")

        if any(t in model_id.lower() for t in ("gguf", "awq", "gptq", "exl2", "nvfp4", "fp8")):
            notes.append("pre-quantized build")

        out.append(
            ModelCandidate(
                model_id=model_id,
                source="hf",
                params=n_params,
                downloads=int(entry.get("downloads") or 0),
                likes=int(entry.get("likes") or 0),
                pipeline_tag=pipeline,
                dtype=dtype,
                tags=list(entry.get("tags") or [])[:12],
                notes=notes,
            )
        )
    return out


def search_ollama(
    query: str = "",
    *,
    limit: int = 20,
    timeout: float = 20.0,
) -> list[ModelCandidate]:
    """List models from the Ollama library HTML index.

    Ollama exposes no JSON catalog, so this reads the library page and extracts
    model names. It is deliberately tolerant: any change in their markup
    degrades to an empty list rather than an exception, because this is a
    convenience source and Hugging Face is the primary one.
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(OLLAMA_LIBRARY_URL)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPError:
        return []

    names = re.findall(r'x-test-search-response[^>]*>\s*([a-z0-9._-]+)\s*<', html)
    if not names:
        names = re.findall(r'href="/library/([a-z0-9._-]+)"', html)

    seen: set[str] = set()
    out: list[ModelCandidate] = []
    q = query.strip().lower()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if q and q not in name.lower():
            continue
        out.append(
            ModelCandidate(
                model_id=name,
                source="ollama",
                params=_params_from_name(name),
                notes=["served through Ollama"],
            )
        )
        if len(out) >= limit:
            break
    return out


def search_all(
    query: str = "",
    *,
    limit: int = 30,
    include_ollama: bool = True,
) -> list[ModelCandidate]:
    """Search every configured catalog and return one deduplicated list."""
    results = search_huggingface(query, limit=limit)
    if include_ollama:
        results.extend(search_ollama(query, limit=max(5, limit // 3)))

    # A model can appear in both catalogs; keep the richer HF record.
    by_id: dict[str, ModelCandidate] = {}
    for candidate in results:
        key = candidate.model_id.lower()
        existing = by_id.get(key)
        if existing is None or candidate.source == "hf":
            by_id[key] = candidate
    return sorted(by_id.values(), key=lambda c: c.downloads, reverse=True)

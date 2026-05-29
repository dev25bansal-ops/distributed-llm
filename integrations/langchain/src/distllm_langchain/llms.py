from typing import Any, AsyncIterator, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult

from distllm.sdk import DistLLMClient, DistLLMClientSync
from distllm.sdk.types import CompletionResponse


class DistLLM(BaseLLM):
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    timeout: float = 120.0

    _client: DistLLMClientSync = None
    _async_client: DistLLMClient = None

    class Config:
        extra = "allow"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = DistLLMClientSync(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )
        self._async_client = DistLLMClient(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )

    @property
    def _llm_type(self) -> str:
        return "distllm-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens) or 256
        model = kwargs.pop("model", self.model)
        resp = self._client.completions(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=kwargs.pop("temperature", self.temperature),
            top_p=kwargs.pop("top_p", self.top_p),
            stop=stop,
        )
        return self._extract_text(resp)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens) or 256
        model = kwargs.pop("model", self.model)
        resp = await self._async_client.completions(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=kwargs.pop("temperature", self.temperature),
            top_p=kwargs.pop("top_p", self.top_p),
            stop=stop,
        )
        return self._extract_text(resp)

    def _generate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens) or 256
        model = kwargs.pop("model", self.model)
        for chunk in self._client.completions_stream(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=kwargs.pop("temperature", self.temperature),
            stop=stop,
        ):
            delta = chunk if isinstance(chunk, dict) else {}
            text = delta.get("choices", [{}])[0].get("text", "")
            if not text:
                continue
            gen_chunk = GenerationChunk(text=text)
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=gen_chunk)
            yield gen_chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens) or 256
        model = kwargs.pop("model", self.model)
        async for chunk in self._async_client.completions_stream(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=kwargs.pop("temperature", self.temperature),
            stop=stop,
        ):
            delta = chunk if isinstance(chunk, dict) else {}
            text = delta.get("choices", [{}])[0].get("text", "")
            if not text:
                continue
            gen_chunk = GenerationChunk(text=text)
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=gen_chunk)
            yield gen_chunk

    @staticmethod
    def _extract_text(resp: Any) -> str:
        if isinstance(resp, CompletionResponse):
            return resp.choices[0].text if resp.choices else ""
        if isinstance(resp, dict):
            return resp.get("choices", [{}])[0].get("text", "")
        return getattr(resp, "choices", [{}])[0].get("text", "")

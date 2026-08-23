"""WAN-optimized speculative decoding.

Combines speculative decoding with WAN token accumulation to minimize
round-trips across high-latency links. Instead of sending one token
per WAN traversal, generates N draft tokens locally and sends them
all for verification in a single round-trip.

WAN round-trip reduction:
  Standard:    1 token/RTT  → 200ms/token at 200ms RTT
  Accumulated: N tokens/RTT → 200ms/N per token
  Speculative: N draft + 1 verify/RTT → even better with high acceptance

Usage::

    decoder = WANSpeculativeDecoder(
        target_forward=remote_pipeline.run_pipeline,
        draft_forward=local_draft_model,
        num_candidates=8,
    )
    output = decoder.generate(input_ids, max_new_tokens=256)
"""


from __future__ import annotations
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from loguru import logger


class WANSpeculativeDecoder:
    """Speculative decoding optimized for WAN links.


    Generates draft tokens locally (fast, on-device), then sends
    all candidates across WAN for batch verification by the target
    model. This reduces WAN round-trips from O(N) to O(N/k) where
    k is the number of candidates per batch.

    Args:
        target_forward: Async callable that runs the target model
            across WAN. Accepts (input_ids, **kwargs) and returns logits.
        draft_forward: Local callable that generates draft tokens.
            Accepts (prefix, num_tokens, **kwargs) and returns token IDs.
        num_candidates: Number of draft tokens per WAN round-trip.
        temperature: Sampling temperature for draft and verification.
        top_k: Top-k sampling for draft generation.
        device: Torch device for local operations.
        max_speculation_depth: Maximum tokens to speculate before
            falling back to standard generation.
    """


    def __init__(
        self,
        target_forward: Callable,
        draft_forward: Callable,
        num_candidates: int = 8,
        temperature: float = 1.0,
        top_k: int = 20,
        device: str = "cuda",
        max_speculation_depth: int = 16,
    ):
        self._target = target_forward
        self._draft = draft_forward
        self._num_candidates = num_candidates
        self._temperature = temperature
        self._top_k = top_k
        self._device = torch.device(device)
        self._max_speculation_depth = max_speculation_depth

        self._stats = {
            "draft_calls": 0,
            "target_calls": 0,
            "tokens_accepted": 0,
            "tokens_rejected": 0,
            "wan_rounds": 0,
            "total_draft_tokens": 0,
        }

    @property
    def stats(self) -> dict:
        """Return speculative decoding statistics."""

        s = dict(self._stats)
        total = s["tokens_accepted"] + s["tokens_rejected"]
        s["acceptance_rate"] = s["tokens_accepted"] / max(total, 1)
        s["wan_speedup"] = (
            s["tokens_accepted"] / max(s["wan_rounds"], 1)
            if s["wan_rounds"] > 0 else 0
        )
        return s

    @torch.no_grad()
    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using WAN speculative decoding.


        Args:
            input_ids: Prompt token IDs, shape ``(1, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments forwarded to forward functions.

        Returns:
            Generated token IDs, shape ``(1, prompt_len + generated)``.
        """

        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            num_draft = min(self._num_candidates, remaining, self._max_speculation_depth)

            if num_draft <= 0:
                break

            # Phase 1: Generate draft tokens locally (fast, no WAN)
            draft_tokens = self._draft_forward(generated, num_draft, **kwargs)
            self._stats["draft_calls"] += 1
            self._stats["total_draft_tokens"] += num_draft

            # Phase 2: Send all draft tokens to WAN for verification (one round-trip)
            full_input = torch.cat([generated, draft_tokens], dim=1)
            t0 = time.monotonic()

            target_logits = await self._target(full_input, **kwargs)
            self._stats["target_calls"] += 1
            self._stats["wan_rounds"] += 1

            wan_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                f"WAN speculative: {num_draft} candidates verified in {wan_ms:.0f}ms "
                f"(RTT amortized over {num_draft} tokens)"
            )

            # Phase 3: Verify each draft token against target distribution
            accepted_count = self._verify_tokens(
                generated, full_input, draft_tokens, target_logits, **kwargs
            )

            # Append accepted tokens
            generated = torch.cat([generated, draft_tokens[:, :accepted_count]], dim=1)
            self._stats["tokens_accepted"] += accepted_count
            self._stats["tokens_rejected"] += num_draft - accepted_count

            if accepted_count < num_draft:
                # Sample correction token from target at rejection point
                next_logits = target_logits[:, generated.shape[1] - 1, :]
                next_token = self._sample(next_logits)
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _draft_forward(
        self, prefix: torch.Tensor, num_tokens: int, **kwargs: Any
    ) -> torch.Tensor:
        """Generate draft tokens locally (fast, on-device)."""

        if num_tokens <= 0:
            return torch.empty(1, 0, dtype=torch.long, device=prefix.device)

        draft_tokens = []
        current = prefix

        for _ in range(num_tokens):
            logits = self._draft(current, **kwargs)
            next_logits = logits[:, -1, :] if logits.dim() > 2 else logits
            token = self._sample_draft(next_logits)
            draft_tokens.append(token)
            current = torch.cat([current, token], dim=1)

        return torch.cat(draft_tokens, dim=1)

    def _verify_tokens(
        self,
        prefix: torch.Tensor,
        full_input: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        **kwargs: Any,
    ) -> int:
        """Verify draft tokens against target model distribution.


        Returns the number of accepted draft tokens.
        """

        num_draft = draft_tokens.shape[1]
        prefix_len = prefix.shape[1] - 1

        if self._temperature == 0:
            # Greedy: accept if argmax matches
            for i in range(num_draft):
                target_argmax = target_logits[:, prefix_len + i, :].argmax(dim=-1).item()
                if target_argmax != draft_tokens[0, i].item():
                    return i
            return num_draft

        # Probabilistic: accept with probability min(1, target_prob/draft_prob)
        for i in range(num_draft):
            target_probs = F.softmax(
                target_logits[:, prefix_len + i, :] / self._temperature, dim=-1
            )
            target_prob = target_probs[0, draft_tokens[0, i].item()].item()

            # Accept if target model assigns high probability
            if target_prob > 0.5:
                continue

            # Stochastic acceptance
            draft_logits = self._draft(
                full_input[:, :prefix_len + i + 1], **kwargs
            )
            draft_probs = F.softmax(
                draft_logits[:, -1, :] / self._temperature, dim=-1
            )
            draft_prob = draft_probs[0, draft_tokens[0, i].item()].item()

            if draft_prob > 0 and torch.rand(1).item() < target_prob / draft_prob:
                continue

            return i

        return num_draft

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from target model logits."""

        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _sample_draft(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from draft model logits (with top-k filtering)."""

        if self._top_k > 0:
            top_k = min(self._top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if self._temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / self._temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class WANSpeculativeConfig:
    """Configuration for WAN speculative decoding."""


    def __init__(
        self,
        num_candidates: int = 8,
        temperature: float = 1.0,
        top_k: int = 20,
        max_speculation_depth: int = 16,
        adaptive_candidates: bool = True,
        min_acceptance_rate: float = 0.3,
    ):
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_k = top_k
        self.max_speculation_depth = max_speculation_depth
        self.adaptive_candidates = adaptive_candidates
        self.min_acceptance_rate = min_acceptance_rate

    def adapt_candidates(self, acceptance_rate: float) -> int:
        """Adaptively adjust candidate count based on acceptance rate.


        High acceptance → more candidates (speculate further)
        Low acceptance → fewer candidates (reduce wasted computation)
        """

        if not self.adaptive_candidates:
            return self.num_candidates

        if acceptance_rate > 0.8:
            return min(self.num_candidates * 2, self.max_speculation_depth)
        elif acceptance_rate > 0.5:
            return self.num_candidates
        elif acceptance_rate > self.min_acceptance_rate:
            return max(2, self.num_candidates // 2)
        else:
            return max(1, self.num_candidates // 4)

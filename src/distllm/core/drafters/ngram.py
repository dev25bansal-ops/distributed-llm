from collections import defaultdict


class NgramMatcher:
    """N-gram based speculative decoding (free draft tokens, no extra model).

    Matches n-grams from the generated text to predict likely next tokens
    based on previously seen sequences. Works well for repetitive text,
    code, and structured outputs.
    """

    def __init__(self, min_match: int = 4, max_match: int = 10):
        self.min_match = min_match
        self.max_match = max_match
        # Maps n-gram tuple -> list of next tokens seen
        self._ngram_index: dict[tuple[int, ...], list[int]] = defaultdict(list)
        self._total_tokens_seen = 0
        self._last_processed = 0

    def update(self, token_ids: list[int]) -> None:
        """Index newly generated tokens for future matching."""
        start = self._last_processed
        self._last_processed = len(token_ids)
        new_tokens = token_ids[start:]
        if not new_tokens:
            return
        max_n = min(self.max_match + 1, len(token_ids) + 1)
        for n in range(self.min_match, max_n):
            for i in range(max(0, start - n), len(token_ids) - n):
                ngram = tuple(token_ids[i : i + n])
                next_token = token_ids[i + n]
                self._ngram_index[ngram].append(next_token)
        self._total_tokens_seen += len(new_tokens)

    def predict(self, context: list[int], max_drafts: int = 5) -> list[int]:
        """Predict draft tokens based on n-gram matching.

        Uses the longest matching n-gram from the end of context
        to find the most likely next tokens.

        Args:
            context: Recent token IDs to match against.
            max_drafts: Maximum number of draft tokens to generate.

        Returns:
            List of predicted draft token IDs.
        """
        if self._total_tokens_seen == 0:
            return []

        drafts = []
        current_context = list(context)

        for _ in range(max_drafts):
            best_match = []
            best_n = 0

            # Try to find longest matching n-gram
            for n in range(min(self.max_match, len(current_context)), self.min_match - 1, -1):
                ngram = tuple(current_context[-n:])
                if ngram in self._ngram_index:
                    candidates = self._ngram_index[ngram]
                    # Weighted sampling for diversity
                    from collections import Counter
                    counts = Counter(candidates)
                    tokens, weights = zip(*counts.most_common(10))
                    total = sum(weights)
                    probs = [w / total for w in weights]
                    import random
                    next_token = random.choices(tokens, weights=probs, k=1)[0]
                    best_match = [next_token]
                    best_n = n
                    break

            if not best_match:
                break

            drafts.extend(best_match)
            current_context.extend(best_match)

        return drafts[:max_drafts]

    def stats(self) -> dict:
        return {
            "total_tokens_indexed": self._total_tokens_seen,
            "unique_ngrams": len(self._ngram_index),
        }

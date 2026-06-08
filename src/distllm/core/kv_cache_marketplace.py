"""KV Cache Marketplace — nodes advertise and trade cached KV states.

Nodes with cached KV states advertise them to the cluster. Other nodes
can discover and purchase cached states for their prompts, reducing
cold-start latency from O(prompt_length) to O(1) for cached prompts.

Architecture:
    Node A caches KV for "What is Python?"
        → Advertises via gossip: (prompt_hash, model, layer_range, price)
    Node B has same prompt
        → Looks up marketplace → finds Node A's advertisement
        → Purchases access → downloads KV cache → skips prefill

Credit economy:
    - Serving nodes earn credits per token served
    - Consuming nodes spend credits per token consumed
    - Credits track contribution to the cluster
"""

from __future__ import annotations

import threading
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CacheAdvertisement:
    """A node's offer to share a cached KV state for a prompt.

    Attributes:
        ad_id: Unique advertisement ID (hash of prompt+model+node).
        prompt_hash: SHA-256 hash of the prompt text (first 16 hex chars).
        model_name: Model this cache is for.
        layer_range: Tuple of (start_layer, end_layer) inclusive.
        token_count: Number of tokens cached.
        node_id: Node serving this cache.
        price_credits: Cost in credits to access this cache.
        created_at: When this advertisement was created.
        ttl_seconds: How long this advertisement is valid.
    """
    ad_id: str
    prompt_hash: str
    model_name: str
    layer_range: tuple[int, int]
    token_count: int
    node_id: str
    price_credits: float = 0.0
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class CacheTransaction:
    """A completed cache purchase."""
    ad_id: str
    buyer_node_id: str
    seller_node_id: str
    tokens_served: int
    credits_paid: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True


class CacheMarketplace:
    """Decentralized marketplace for KV cache advertisements and purchases.

    Nodes advertise cached prompts. Other nodes discover and purchase
    access to these caches. Credits track contribution — nodes that
    serve more earn credits to spend later.

    Thread-safe: all public methods acquire self._lock.
    """

    def __init__(
        self,
        node_id: str = "coordinator",
        default_price_per_token: float = 0.001,
        ad_ttl_seconds: float = 3600.0,
        reputation_system: Any = None,
    ):
        self._node_id = node_id
        self._default_price = default_price_per_token
        self._ad_ttl = ad_ttl_seconds
        self._reputation = reputation_system

        # ad_id -> CacheAdvertisement
        self._advertisements: dict[str, CacheAdvertisement] = {}
        # node_id -> credit balance
        self._credit_balances: dict[str, float] = {}
        # Transaction log
        self._transactions: list[CacheTransaction] = []

        self._lock = threading.Lock()
        self._last_cleanup: float = time.monotonic()

        # Stats
        self._hits = 0
        self._misses = 0
        self._total_servings = 0

    # ── Advertisement Management ──────────────────────────────────────────

    def advertise(
        self,
        prompt: str,
        model_name: str,
        layer_range: tuple[int, int],
        token_count: int,
        node_id: str | None = None,
        price_credits: float | None = None,
    ) -> str:
        """Advertise a cached KV state for a prompt.

        Returns the advertisement ID. Other nodes can discover and
        purchase access to this cache via lookup() + purchase().
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        node = node_id or self._node_id
        ad_id = f"{prompt_hash}-{model_name.replace('/', '_')}-{node}"

        ad = CacheAdvertisement(
            ad_id=ad_id,
            prompt_hash=prompt_hash,
            model_name=model_name,
            layer_range=layer_range,
            token_count=token_count,
            node_id=node,
            price_credits=price_credits or (token_count * self._default_price),
            ttl_seconds=self._ad_ttl,
        )

        with self._lock:
            self._advertisements[ad_id] = ad

        logger.debug(f"KV cache advertised: {ad_id} ({token_count} tokens, {price_credits} credits)")
        return ad_id

    def revoke_advertisement(self, ad_id: str) -> bool:
        """Remove an advertisement (e.g., cache was evicted)."""
        with self._lock:
            return self._advertisements.pop(ad_id, None) is not None

    def get_advertisement(self, ad_id: str) -> CacheAdvertisement | None:
        with self._lock:
            return self._advertisements.get(ad_id)

    # ── Discovery ─────────────────────────────────────────────────────────

    def lookup(
        self,
        prompt_hash: str,
        model_name: str,
        max_results: int = 5,
    ) -> list[CacheAdvertisement]:
        """Find cache advertisements for a prompt.

        Results are sorted by:
        1. Token count (higher = more valuable cache)
        2. Price (lower = cheaper)
        3. Node reputation (higher = more trusted)

        Returns empty list if no matching advertisements found.
        """
        with self._lock:
            matching: list[CacheAdvertisement] = []
            for ad in self._advertisements.values():
                if ad.is_expired:
                    continue
                if ad.prompt_hash == prompt_hash and ad.model_name == model_name:
                    matching.append(ad)

            # Sort: more tokens first, then cheaper, then higher reputation
            def _sort_key(ad: CacheAdvertisement) -> tuple:
                rep_score = 0.0
                if self._reputation is not None:
                    record = self._reputation.get_record(ad.node_id)
                    if record:
                        rep_score = record.reliability
                return (-ad.token_count, ad.price_credits, -rep_score)

            matching.sort(key=_sort_key)
            self._hits += 1 if matching else 0
            self._misses += 0 if matching else 1
            return matching[:max_results]

    def lookup_by_node(self, node_id: str) -> list[CacheAdvertisement]:
        """Find all advertisements from a specific node."""
        with self._lock:
            return [
                ad for ad in self._advertisements.values()
                if ad.node_id == node_id and not ad.is_expired
            ]

    # ── Purchase / Credit Accounting ──────────────────────────────────────

    def purchase(self, ad_id: str, buyer_node_id: str) -> bool:
        """Purchase access to a cached KV advertisement.

        Checks that:
        - The advertisement still exists and is not expired
        - The buyer has sufficient credits

        On success, records a transaction and debits the buyer's credits.
        Returns True if the purchase was successful.
        """
        with self._lock:
            ad = self._advertisements.get(ad_id)
            if ad is None:
                logger.warning(f"Purchase failed: ad {ad_id} not found")
                return False
            if ad.is_expired:
                logger.warning(f"Purchase failed: ad {ad_id} expired")
                self._advertisements.pop(ad_id, None)
                return False

            buyer_balance = self._credit_balances.get(buyer_node_id, 0.0)
            if buyer_balance < ad.price_credits:
                logger.warning(
                    f"Purchase failed: {buyer_node_id} has {buyer_balance:.2f} credits, "
                    f"needs {ad.price_credits:.2f}"
                )
                return False

            # Debit buyer, credit seller
            self._credit_balances[buyer_node_id] = buyer_balance - ad.price_credits
            seller_balance = self._credit_balances.get(ad.node_id, 0.0)
            self._credit_balances[ad.node_id] = seller_balance + ad.price_credits

            self._total_servings += 1

            tx = CacheTransaction(
                ad_id=ad_id,
                buyer_node_id=buyer_node_id,
                seller_node_id=ad.node_id,
                tokens_served=ad.token_count,
                credits_paid=ad.price_credits,
            )
            self._transactions.append(tx)

            logger.debug(
                f"KV cache purchase: {buyer_node_id} bought {ad_id} "
                f"for {ad.price_credits} credits"
            )
            return True

    def record_served_tokens(self, node_id: str, token_count: int) -> None:
        """Credit a node for serving cached tokens."""
        with self._lock:
            earnings = token_count * self._default_price
            current = self._credit_balances.get(node_id, 0.0)
            self._credit_balances[node_id] = current + earnings

    def record_consumed_tokens(self, node_id: str, token_count: int) -> None:
        """Debit a node for consuming cached tokens."""
        with self._lock:
            cost = token_count * self._default_price
            current = self._credit_balances.get(node_id, 0.0)
            self._credit_balances[node_id] = max(0.0, current - cost)

    def get_balance(self, node_id: str) -> float:
        """Get a node's credit balance."""
        with self._lock:
            return self._credit_balances.get(node_id, 0.0)

    # ── Cleanup ───────────────────────────────────────────────────────────

    def expire_stale_ads(self) -> int:
        """Remove expired advertisements. Returns count removed."""
        now = time.monotonic()
        if now - self._last_cleanup < 60:
            return 0
        self._last_cleanup = now

        expired = []
        with self._lock:
            for ad_id, ad in list(self._advertisements.items()):
                if ad.is_expired:
                    expired.append(ad_id)
                    del self._advertisements[ad_id]
        if expired:
            logger.debug(f"Expired {len(expired)} stale cache advertisements")
        return len(expired)

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "active_advertisements": len(self._advertisements),
                "total_transactions": len(self._transactions),
                "total_servings": self._total_servings,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "credits_in_circulation": round(sum(self._credit_balances.values()), 2),
                "active_nodes": len(set(
                    ad.node_id for ad in self._advertisements.values()
                )),
            }

    def get_node_for_cached_prompt(
        self, prompt_hash: str, model_name: str,
    ) -> str | None:
        """Quick lookup: find a node serving cached KV for this prompt.

        Useful for the coordinator to skip prefill and directly route
        to a node that already has the prefix cached.
        """
        ads = self.lookup(prompt_hash, model_name, max_results=1)
        return ads[0].node_id if ads else None

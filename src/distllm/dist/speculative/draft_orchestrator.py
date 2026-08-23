"""
Draft orchestrator for heterogeneous draft-model fleet management.

Uses Thompson sampling (Beta-Bernoulli bandit) to select optimal subsets
of draft models per request domain, balancing exploration vs exploitation
while respecting latency budgets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Thompson sampling bandit — Beta-Bernoulli per (draft_id, domain) pair
# ---------------------------------------------------------------------------

class ThompsonSamplingBandit:
    """Beta-Bernoulli Thompson-sampling bandit over draft-model / domain pairs.

    Each (draft_id, domain) combination maintains an independent Beta posterior
    over the latent acceptance rate.  At selection time we draw one sample from
    each candidate's Beta posterior and pick the top-k by sampled value, which
    naturally balances exploration (high variance = more aggressive draws) and
    exploitation (high mean = more likely to be selected).

    Parameters
    ----------
    alpha_prior : float
        Prior pseudo-count for accepted tokens (default 1.0 — uniform prior).
    beta_prior : float
        Prior pseudo-count for rejected tokens (default 1.0 — uniform prior).
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> None:
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("Alpha and beta priors must be positive")
        self._alpha_prior = alpha_prior
        self._beta_prior = beta_prior
        # Nested dict: self._params[draft_id][domain] = (alpha, beta)
        self._params: dict[str, dict[str, tuple[float, float]]] = {}

    # -- Public API ------------------------------------------------------------

    def select(
        self, draft_ids: list[str], domain: str, k: int = 1
    ) -> list[str]:
        """Select *k* draft models via Thompson sampling.

        For each candidate draft we draw a single sample from its Beta posterior
        (or the prior if unseen) and return the *k* draft ids with the highest
        sampled values.

        Parameters
        ----------
        draft_ids : list of str
            Candidate draft model identifiers.
        domain : str
            Request domain (e.g. ``"code"``, ``"math"``, ``"general"``).
        k : int
            Number of drafts to select (default 1, clamped to ``len(draft_ids)``).

        Returns
        -------
        list of str
            Up to *k* draft ids sorted descending by sampled acceptance rate.
        """
        if k < 1:
            return []
        k = min(k, len(draft_ids))
        if k == 0:
            return []

        samples: list[tuple[str, float]] = []
        for did in draft_ids:
            alpha, beta = self._get_params(did, domain)
            # Sample from Beta(alpha, beta)
            theta = random.betavariate(alpha, beta)
            samples.append((did, theta))

        samples.sort(key=lambda x: x[1], reverse=True)
        return [did for did, _ in samples[:k]]

    def update(
        self, draft_id: str, domain: str, accepted: int, rejected: int
    ) -> None:
        """Update the Beta posterior with observed token counts.

        Parameters
        ----------
        draft_id : str
            Draft model identifier.
        domain : str
            Request domain.
        accepted : int
            Number of accepted draft tokens.
        rejected : int
            Number of rejected draft tokens.
        """
        if accepted < 0 or rejected < 0:
            raise ValueError("Accepted and rejected counts must be non-negative")
        alpha, beta = self._get_params(draft_id, domain)
        self._set_params(draft_id, domain, alpha + accepted, beta + rejected)

    def get_acceptance_rate(self, draft_id: str, domain: str) -> float:
        """Return the posterior mean acceptance rate for a (draft, domain) pair.

        Parameters
        ----------
        draft_id : str
            Draft model identifier.
        domain : str
            Request domain.

        Returns
        -------
        float
            Mean of the Beta posterior (i.e. ``alpha / (alpha + beta)``).
            Returns the prior mean if no observations exist yet.
        """
        alpha, beta = self._get_params(draft_id, domain)
        total = alpha + beta
        return alpha / total if total > 0 else self._alpha_prior / (
            self._alpha_prior + self._beta_prior
        )

    # -- Internal helpers ------------------------------------------------------

    def _get_params(self, draft_id: str, domain: str) -> tuple[float, float]:
        """Return the current (alpha, beta) for a (draft, domain) pair.

        Falls back to the prior if no observations have been recorded.
        """
        domain_params = self._params.get(draft_id)
        if domain_params is None:
            return (self._alpha_prior, self._beta_prior)
        return domain_params.get(domain, (self._alpha_prior, self._beta_prior))

    def _set_params(
        self, draft_id: str, domain: str, alpha: float, beta: float
    ) -> None:
        """Store updated (alpha, beta) for a (draft, domain) pair."""
        if draft_id not in self._params:
            self._params[draft_id] = {}
        self._params[draft_id][domain] = (alpha, beta)


# ---------------------------------------------------------------------------
# Persistence-friendly acceptance matrix
# ---------------------------------------------------------------------------

class DomainAcceptanceMatrix:
    """Container for acceptance statistics with serialisation and decay support.

    Internally stores ``{(draft_id, domain): (alpha, beta)}`` and exposes
    dictionary round-trip for checkpointing / persistence.

    Parameters
    ----------
    alpha_prior : float
        Prior alpha used when constructing fresh entries (default 1.0).
    beta_prior : float
        Prior beta used when constructing fresh entries (default 1.0).
    """

    def __init__(
        self, alpha_prior: float = 1.0, beta_prior: float = 1.0
    ) -> None:
        self._alpha_prior = alpha_prior
        self._beta_prior = beta_prior
        self._data: dict[tuple[str, str], tuple[float, float]] = {}

    # -- Accessors -------------------------------------------------------------

    def get(
        self, draft_id: str, domain: str
    ) -> tuple[float, float]:
        """Return ``(alpha, beta)`` for the given key, falling back to prior.

        Parameters
        ----------
        draft_id : str
            Draft model identifier.
        domain : str
            Request domain.

        Returns
        -------
        tuple of (float, float)
            The stored (alpha, beta) or the prior if unseen.
        """
        return self._data.get(
            (draft_id, domain), (self._alpha_prior, self._beta_prior)
        )

    def set(
        self, draft_id: str, domain: str, alpha: float, beta: float
    ) -> None:
        """Set (alpha, beta) for a given key.

        Parameters
        ----------
        draft_id : str
            Draft model identifier.
        domain : str
            Request domain.
        alpha : float
            Alpha parameter (must be > 0).
        beta : float
            Beta parameter (must be > 0).
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"Alpha ({alpha}) and beta ({beta}) must be positive"
            )
        self._data[(draft_id, domain)] = (alpha, beta)

    def items(self) -> list[tuple[str, str, float, float]]:
        """Iterate over all stored entries.

        Returns
        -------
        list of (draft_id, domain, alpha, beta)
        """
        return [
            (did, dom, a, b)
            for (did, dom), (a, b) in self._data.items()
        ]

    @property
    def entry_count(self) -> int:
        """Total number of stored (draft_id, domain) entries."""
        return len(self._data)

    # -- Serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the matrix to a JSON-safe dictionary.

        Returns
        -------
        dict
            Structure::

                {
                    "alpha_prior": 1.0,
                    "beta_prior": 1.0,
                    "entries": {
                        "<draft_id>::<domain>": [alpha, beta],
                        ...
                    }
                }
        """
        entries = {}
        for (did, dom), (a, b) in self._data.items():
            entries[f"{did}::{dom}"] = [a, b]
        return {
            "alpha_prior": self._alpha_prior,
            "beta_prior": self._beta_prior,
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainAcceptanceMatrix:
        """Deserialize from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialized matrix dictionary.

        Returns
        -------
        DomainAcceptanceMatrix
            Reconstructed instance.
        """
        alpha_prior = data.get("alpha_prior", 1.0)
        beta_prior = data.get("beta_prior", 1.0)
        matrix = cls(alpha_prior=alpha_prior, beta_prior=beta_prior)

        for key, (a, b) in data.get("entries", {}).items():
            if "::" not in key:
                continue
            draft_id, domain = key.split("::", 1)
            matrix._data[(draft_id, domain)] = (float(a), float(b))

        return matrix

    # -- Decay -----------------------------------------------------------------

    def decay(self, rate: float = 0.99) -> None:
        """Apply exponential decay to all stored counts.

        Multiplying both alpha and beta by *rate* reduces the weight of older
        observations, allowing the bandit to adapt to changing conditions.

        Parameters
        ----------
        rate : float
            Decay multiplier in ``(0, 1]`` (default 0.99).  A value of 1.0
            applies no decay.
        """
        if not 0 < rate <= 1:
            raise ValueError(f"Decay rate must be in (0, 1], got {rate}")
        for key in list(self._data.keys()):
            a, b = self._data[key]
            decayed = (a * rate, b * rate)
            # Prune entries that have effectively vanished
            if decayed[0] < 1e-12 and decayed[1] < 1e-12:
                del self._data[key]
            else:
                self._data[key] = decayed


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DraftModelInfo:
    """Metadata for a registered draft model.

    Attributes
    ----------
    draft_id : str
        Unique identifier.
    model_name : str
        HuggingFace or local model name/path.
    hardware : str
        Hardware label (e.g. ``"A100-40GB"``, ``"RTX-4090"``).
    cost_per_token : float
        Inference cost per draft token in arbitrary units.
    total_calls : int
        Cumulative number of times this draft has been selected.
    total_accepted : int
        Cumulative accepted tokens across all invocations.
    total_rejected : int
        Cumulative rejected tokens across all invocations.
    """

    draft_id: str
    model_name: str
    hardware: str
    cost_per_token: float
    total_calls: int = 0
    total_accepted: int = 0
    total_rejected: int = 0


# Simple type for a callable that extracts a domain label from raw text.
DomainRouter = Any  # duck-typed: callable[[str], str]


class DraftOrchestrator:
    """Manages a heterogeneous fleet of draft models and selects optimal subsets.

    Uses a :class:`ThompsonSamplingBandit` to choose drafts per domain and
    optionally leverages an *agentic router* to classify request text into
    a domain label.

    Parameters
    ----------
    draft_bank : dict of str -> DraftModelInfo
        Pre-populated dictionary of available draft models keyed by ``draft_id``.
    agentic_router : callable or None
        Optional callable ``(request_text: str) -> domain: str``.  When provided,
        :meth:`select_drafts` will use it to estimate the domain from the request
        text automatically.

    Examples
    --------
    >>> orch = DraftOrchestrator(draft_bank={})
    >>> orch.register_draft("fast-v0", "model-small", "RTX-4090", 0.01)
    >>> orch.register_draft("big-v1", "model-large", "A100-80GB", 0.05)
    >>> chosen = orch.select_drafts("Write a quicksort", domain="code", k=2)
    >>> orch.report_outcome("fast-v0", "code", accepted=42, rejected=3)
    """

    def __init__(
        self,
        draft_bank: dict[str, DraftModelInfo] | None = None,
        agentic_router: DomainRouter | None = None,
    ) -> None:
        self._draft_bank: dict[str, DraftModelInfo] = (
            dict(draft_bank) if draft_bank else {}
        )
        self._agentic_router = agentic_router
        self._bandit = ThompsonSamplingBandit()
        self._matrix = DomainAcceptanceMatrix()

    # -- Registration ----------------------------------------------------------

    def register_draft(
        self,
        draft_id: str,
        model_name: str,
        hardware: str,
        cost_per_token: float,
    ) -> None:
        """Register a new draft model in the fleet.

        Parameters
        ----------
        draft_id : str
            Unique identifier for this draft model.
        model_name : str
            Model name or path (e.g. ``"JackFram/llama-68m"``).
        hardware : str
            Hardware descriptor (e.g. ``"A100-40GB"``).
        cost_per_token : float
            Inference cost per draft token (arbitrary unit).
        """
        if draft_id in self._draft_bank:
            raise ValueError(f"Draft model '{draft_id}' is already registered")
        if cost_per_token < 0:
            raise ValueError("cost_per_token must be non-negative")
        self._draft_bank[draft_id] = DraftModelInfo(
            draft_id=draft_id,
            model_name=model_name,
            hardware=hardware,
            cost_per_token=cost_per_token,
        )

    # -- Selection -------------------------------------------------------------

    def select_drafts(
        self,
        request_text: str,
        domain: str | None = None,
        k: int = 2,
        latency_budget_ms: float | None = None,
    ) -> list[str]:
        """Select the best draft models for a given request.

        If *domain* is ``None`` and an ``agentic_router`` was provided, the
        domain is inferred from *request_text*.  When a ``latency_budget_ms``
        is given, candidates whose expected latency (estimated from hardware)
        exceeds the budget are excluded before bandit selection.

        Parameters
        ----------
        request_text : str
            The raw request text (used for domain inference if needed).
        domain : str or None
            Explicit domain label.  If ``None``, the agentic router is used
            (raises ``ValueError`` if no router is available).
        k : int
            Number of drafts to select (default 2).
        latency_budget_ms : float or None
            Maximum allowed per-draft latency in milliseconds.  Drafts whose
            estimated latency exceeds this are filtered out.

        Returns
        -------
        list of str
            Selected draft model IDs (length ≤ *k*).
        """
        # Resolve domain
        if domain is None:
            if self._agentic_router is None:
                raise ValueError(
                    "No domain provided and no agentic_router configured"
                )
            domain = self._agentic_router(request_text)

        # Gather candidate draft ids
        candidate_ids = list(self._draft_bank.keys())
        if not candidate_ids:
            return []

        # Apply latency filter
        if latency_budget_ms is not None and latency_budget_ms > 0:
            candidate_ids = [
                did
                for did in candidate_ids
                if self._estimate_latency_ms(did) <= latency_budget_ms
            ]
            if not candidate_ids:
                return []

        # Thompson-sample
        return self._bandit.select(candidate_ids, domain, k=k)

    # -- Outcome reporting -----------------------------------------------------

    def report_outcome(
        self,
        draft_id: str,
        domain: str,
        accepted_tokens: int,
        rejected_tokens: int,
    ) -> None:
        """Report a draft outcome to update the bandit and internal stats.

        Parameters
        ----------
        draft_id : str
            Draft model identifier.
        domain : str
            Request domain this outcome applies to.
        accepted_tokens : int
            Number of tokens accepted from the draft.
        rejected_tokens : int
            Number of tokens rejected from the draft.
        """
        # Update bandit posterior
        self._bandit.update(draft_id, domain, accepted_tokens, rejected_tokens)

        # Persist into acceptance matrix
        cur_a, cur_b = self._matrix.get(draft_id, domain)
        self._matrix.set(
            draft_id,
            domain,
            cur_a + accepted_tokens,
            cur_b + rejected_tokens,
        )

        # Update aggregate stats
        info = self._draft_bank.get(draft_id)
        if info is not None:
            info.total_calls += 1
            info.total_accepted += accepted_tokens
            info.total_rejected += rejected_tokens

    # -- Fleet introspection ---------------------------------------------------

    def get_fleet_status(self) -> dict[str, Any]:
        """Return a snapshot of the entire draft fleet.

        Returns
        -------
        dict
            Structure::

                {
                    "drafts": {
                        "<draft_id>": {
                            "model_name": ...,
                            "hardware": ...,
                            "cost_per_token": ...,
                            "total_calls": ...,
                            "total_accepted": ...,
                            "total_rejected": ...,
                            "acceptance_rate": ...,
                        },
                        ...
                    },
                    "total_drafts": ...,
                    "matrix_entries": ...,
                }

        The ``acceptance_rate`` is the average over all observed domains for
        that draft (or ``None`` if no observations exist).
        """
        drafts: dict[str, Any] = {}
        for did, info in self._draft_bank.items():
            # Compute per-draft average acceptance rate across domains
            rates: list[float] = []
            for (dd, _dom), (a, b) in self._matrix._data.items():
                if dd == did:
                    total = a + b
                    if total > 0:
                        rates.append(a / total)
            avg_rate = sum(rates) / len(rates) if rates else None

            drafts[did] = {
                "model_name": info.model_name,
                "hardware": info.hardware,
                "cost_per_token": info.cost_per_token,
                "total_calls": info.total_calls,
                "total_accepted": info.total_accepted,
                "total_rejected": info.total_rejected,
                "acceptance_rate": avg_rate,
            }

        return {
            "drafts": drafts,
            "total_drafts": len(drafts),
            "matrix_entries": self._matrix.entry_count,
        }

    # -- Domain router helpers -------------------------------------------------

    @property
    def agentic_router(self) -> DomainRouter | None:
        """The domain router callable, if any."""
        return self._agentic_router

    @agentic_router.setter
    def agentic_router(self, router: DomainRouter | None) -> None:
        """Set or clear the domain router."""
        self._agentic_router = router

    # -- Direct bandit / matrix access -----------------------------------------

    @property
    def bandit(self) -> ThompsonSamplingBandit:
        """The underlying Thompson-sampling bandit instance."""
        return self._bandit

    @property
    def matrix(self) -> DomainAcceptanceMatrix:
        """The underlying domain-acceptance matrix instance."""
        return self._matrix

    @property
    def draft_bank(self) -> dict[str, DraftModelInfo]:
        """Read-only view of the registered draft models."""
        return dict(self._draft_bank)

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _estimate_latency_ms(draft_id: str) -> float:
        """Heuristic latency estimate for a draft model.

        Currently returns a flat placeholder (50 ms).  A production
        implementation should use profiled hardware lookups or a
        cost-model predictor.

        Parameters
        ----------
        draft_id : str
            Draft model identifier (ignored in the base heuristic).

        Returns
        -------
        float
            Estimated latency in milliseconds.
        """
        # TODO: replace with real hardware/model latency table
        _ = draft_id
        return 50.0

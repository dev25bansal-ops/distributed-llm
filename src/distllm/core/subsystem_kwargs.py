"""Startup-kwargs validator for optional subsystems.

The :class:`distllm.core.coordinator.Coordinator` starts ~9 optional
subsystems (``discovery``, ``federation``, ``smart_router``, etc.) via
the ``_start_subsystem`` helper, each receiving a dict of
``constructor_kwargs``.  Historically, every coordinator release adds
a new kwarg to a subsystem's ``__init__`` and the coordinator passes
it — the subsystems grew tolerant of unknown kwargs (e.g. ``min_savings_pct``,
``check_interval_s`` on :class:`ArbitrageEngine`) by accepting them as
``None``-default aliases.  Without the alias, ``__init__`` raised
``TypeError`` and the subsystem silently failed to start.  This was the
**3rd** regression of the exact same shape.

This helper provides a single place to defensively strip unknown
kwargs (with a warning) so a new release can add an arbitrary kwarg
without breaking subsystem startup.  The :func:`validate_subsystem_kwargs`
function takes a class and a kwargs dict, drops any kwarg not in the
class's ``__init__`` signature, logs a warning for each dropped kwarg,
and returns the pruned dict.

Usage in coordinator::

    from distllm.core.subsystem_kwargs import validate_subsystem_kwargs
    cls = getattr(__import__(module_path, fromlist=[class_name]), class_name)
    pruned = validate_subsystem_kwargs(cls, constructor_kwargs or {})
    instance = cls(**pruned)

The warning is loud so operators can see when a coordinator upgrade
silently drops a kwarg — the right fix is to add the kwarg to the
subsystem's signature (or accept it as an alias), but startup no
longer fails while the fix is being written.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def _init_params(cls: type) -> set[str]:
    """Return the set of formal parameter names for ``cls.__init__``.

    Falls back to ``set()`` when introspection fails (e.g. builtins or
    classes with ``__init__`` defined dynamically) — in that case the
    caller will pass all kwargs through unchanged.
    """
    try:
        import inspect

        sig = inspect.signature(cls.__init__)
        return {name for name in sig.parameters if name != "self"}
    except (TypeError, ValueError):
        return set()


def validate_subsystem_kwargs(
    cls: type,
    kwargs: dict[str, Any],
    *,
    subsystem_name: str | None = None,
) -> dict[str, Any]:
    """Strip unknown kwargs with a warning, return the pruned dict.

    Args:
        cls: The subsystem class whose ``__init__`` signature is the
            source of truth for accepted kwargs.
        kwargs: The constructor kwargs to validate.
        subsystem_name: Optional human-readable name for the warning
            (e.g. ``"arbitrage_engine"``).  Falls back to ``cls.__name__``.

    Returns:
        A new dict containing only kwargs that appear in ``cls.__init__``'s
        signature.  The input ``kwargs`` is not mutated.
    """
    if not kwargs:
        return dict(kwargs)

    accepted = _init_params(cls)
    if not accepted:
        # Introspection failed — pass everything through unchanged so
        # the subsystem still has a chance to construct.
        return dict(kwargs)

    label = subsystem_name or cls.__name__
    pruned: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in accepted:
            pruned[key] = value
        else:
            logger.warning(
                "{}: dropping unknown startup kwarg {}={!r} (not in __init__); "
                "add it to the class signature or stop passing it from the coordinator",
                label, key, value,
            )
    return pruned


__all__ = ["validate_subsystem_kwargs"]
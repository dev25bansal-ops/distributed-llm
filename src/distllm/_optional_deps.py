"""Lazy import for optional dependencies.

The tree has many modules that import an optional package (torch,
transformers, ...) at module scope because they use it inside methods. That
turns every transitively-imported module into a hard requirement for the
whole package: `import distllm.api` — the distllm-api console script, the
`distllm run` command — crashed on a torch-free install because some deep
module imported torch at the top.

``optional_import("torch")`` returns the real module when installed, and
otherwise a proxy object that raises RuntimeError naming the package and the
pip command on first use. Module load then succeeds; only code paths that
genuinely need the dependency fail, and they fail with a message that says
what to install.
"""

from __future__ import annotations

import importlib
from typing import Any

# package name -> extra that carries it, for the error message.
_PACKAGE_EXTRAS: dict[str, str] = {
    "torch": "self-hosted",
    "torch.nn": "self-hosted",
    "torch.nn.functional": "self-hosted",
    "torch.distributed": "self-hosted",
    "torchvision": "self-hosted",
    "transformers": "self-hosted",
    "bitsandbytes": "model-extras",
    "peft": "model-extras",
    "vllm": "backends",
    "sglang": "backends",
    "llama_cpp": "backends",
    "ray": "ray",
}


class MissingOptionalDependency:
    """Proxy that raises on any attribute access.

    Raising on *access* (rather than at import) is what makes the module
    importable while keeping misuse loud. ``torch.cuda.is_available()`` on a
    torch-free install raises here instead of an AttributeError on None.
    """

    def __init__(self, package: str) -> None:
        # Instance attributes go through object.__setattr__ because this
        # class overrides __getattr__; __getattr__ only fires for attributes
        # NOT found normally, so plain assignment is fine — but keep it
        # explicit for clarity.
        object.__setattr__(self, "_package", package)

    def __getattr__(self, name: str) -> Any:
        package = object.__getattribute__(self, "_package")
        extra = _PACKAGE_EXTRAS.get(package)
        hint = (
            f"pip install 'distributed-llm[{extra}]'"
            if extra
            else f"pip install {package}"
        )
        raise RuntimeError(
            f"{package} is required here but is not installed. Install it: {hint}"
        )

    def __repr__(self) -> str:
        return f"<missing optional dependency {object.__getattribute__(self, '_package')!r}>"


def optional_import(module_name: str) -> Any:
    """Return ``module_name`` if importable, else a MissingOptionalDependency."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return MissingOptionalDependency(module_name)
    except Exception:
        # A submodule whose import raises something else (e.g. a CUDA
        # assertion during torch install probing) is treated as absent for
        # import purposes; callers that reach real code will surface the
        # original failure through the proxy's message being less
        # specific than ideal, but module import must not die.
        return MissingOptionalDependency(module_name)


def resolve_class(module_name: str, qualname: str) -> Any:
    """Resolve ``module_name.qualname`` or return a stub base class.

    Class *bases* are evaluated at import even under ``from __future__ import
    annotations``, so ``class Foo(torch.nn.Module)`` would evaluate the proxy
    and raise at import even after the import itself was deferred. This
    returns the real class when the module is installed, and otherwise a stub
    that can be subclassed freely — the error surfaces when the class is
    instantiated, with the same install hint.
    """
    module = optional_import(module_name)
    if not isinstance(module, MissingOptionalDependency):
        try:
            return getattr(module, qualname)
        except AttributeError:
            pass

    class _MissingDependencyBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                f"{module_name}.{qualname} is required but "
                f"{module_name.split('.')[0]} is not installed. "
                f"Install it: pip install "
                f"'distributed-llm[{_PACKAGE_EXTRAS.get(module_name.split('.')[0], module_name.split('.')[0])}]'"
            )

    _MissingDependencyBase.__name__ = f"_Missing{qualname}Base"
    _MissingDependencyBase.__qualname__ = _MissingDependencyBase.__name__
    return _MissingDependencyBase


def no_grad():
    """``torch.no_grad()`` when torch is installed, else a no-op.

    Decorators and context managers evaluate at *definition* time, so
    ``@torch.no_grad()`` on a method would evaluate the proxy at import even
    with the import itself deferred. Without torch there are no tensors and
    nothing to disable grad for, so the no-op is the correct semantics, not a
    fallback — the same shape as the no-op tracer in observability/spans.py.
    """
    module = optional_import("torch")
    if isinstance(module, MissingOptionalDependency):
        return _NullContext()
    return module.no_grad()


class _NullContext:
    """Context manager and decorator that does nothing."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False

    def __call__(self, fn: Any) -> Any:
        return fn

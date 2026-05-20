from __future__ import annotations

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def register_split_op() -> Callable[[F], F]:
    """Compatibility decorator (no-op in minisgl)."""

    def decorator(func: F) -> F:
        setattr(func, "__minisgl_split_op__", True)
        return func

    return decorator


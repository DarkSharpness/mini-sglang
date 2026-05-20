from __future__ import annotations

from typing import Any, Callable, Iterable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def register_custom_op(
    *,
    mutates_args: Iterable[str] | None = None,
) -> Callable[[F], F]:
    """Compatibility decorator for custom ops (no-op in minisgl)."""

    def decorator(func: F) -> F:
        setattr(func, "__minisgl_custom_op__", True)
        setattr(func, "__minisgl_mutates_args__", tuple(mutates_args or ()))
        return func

    return decorator


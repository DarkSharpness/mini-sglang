from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class ForwardContext:
    forward_batch: Any
    attention_layers: list[Any]


_FORWARD_CONTEXT: ContextVar[ForwardContext | None] = ContextVar(
    "minisgl_forward_context", default=None
)


def get_forward_context() -> ForwardContext | None:
    return _FORWARD_CONTEXT.get()


@contextmanager
def set_forward_context(
    *,
    forward_batch: Any,
    attention_layers: list[Any],
) -> Iterator[None]:
    token = _FORWARD_CONTEXT.set(
        ForwardContext(
            forward_batch=forward_batch,
            attention_layers=attention_layers,
        )
    )
    try:
        yield
    finally:
        _FORWARD_CONTEXT.reset(token)

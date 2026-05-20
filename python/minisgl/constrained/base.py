from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeAlias

import torch

GrammarKey: TypeAlias = tuple[str, str]


class BaseGrammarObject:
    def __init__(self) -> None:
        self._finished = False
        self.current_token: int | None = None

    def accept_token(self, token: int) -> None:
        raise NotImplementedError()

    def rollback(self, k: int) -> None:
        raise NotImplementedError()

    def is_terminated(self) -> bool:
        return False

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        raise NotImplementedError()

    def copy(self) -> BaseGrammarObject:
        return self

    @property
    def finished(self) -> bool:
        return self._finished

    @finished.setter
    def finished(self, finished: bool) -> None:
        self._finished = finished


GrammarFuture: TypeAlias = Future[BaseGrammarObject | None]
GrammarValue: TypeAlias = BaseGrammarObject | GrammarFuture


INVALID_GRAMMAR_OBJ = BaseGrammarObject()


class BaseGrammarBackend:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor()
        self.cache: dict[GrammarKey, BaseGrammarObject] = {}

    def dispatch_json(self, key_string: str) -> BaseGrammarObject | None:
        raise NotImplementedError()

    def _init_value_dispatch(self, key: GrammarKey) -> BaseGrammarObject | None:
        key_type, key_string = key
        if key_type == "json":
            return self.dispatch_json(key_string)
        raise NotImplementedError(f"Structured output format is not implemented: {key_type}")

    def get_cached_or_future_value(self, key: GrammarKey) -> tuple[GrammarValue, bool]:
        value = self.cache.get(key)
        if value is not None:
            return value.copy(), True
        return self.executor.submit(self._init_value_dispatch, key), False

    def set_cache(self, key: GrammarKey, value: BaseGrammarObject) -> None:
        self.cache[key] = value

    def reset(self) -> None:
        self.cache.clear()

    def shutdown(self) -> None:
        self.reset()
        self.executor.shutdown(wait=False, cancel_futures=True)


def create_grammar_backend(
    tokenizer, vocab_size: int, eos_token_ids: int | list[int] | set[int] | None = None
) -> BaseGrammarBackend | None:
    from .reasoner_backend import ReasonerGrammarBackend
    from .xgrammar_backend import XGrammarBackend

    try:
        backend: BaseGrammarBackend = XGrammarBackend(tokenizer, vocab_size, eos_token_ids)
    except Exception:
        return None
    think_end_id = getattr(tokenizer, "think_end_id", None)
    if think_end_id is not None:
        backend = ReasonerGrammarBackend(backend, think_end_id)
    return backend

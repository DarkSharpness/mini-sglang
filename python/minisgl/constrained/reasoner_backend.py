from __future__ import annotations

import torch

from .base import INVALID_GRAMMAR_OBJ, BaseGrammarBackend, BaseGrammarObject


class ReasonerGrammarObject(BaseGrammarObject):
    def __init__(self, grammar: BaseGrammarObject, think_end_id: int) -> None:
        super().__init__()
        self.grammar = grammar
        self.think_end_id = think_end_id
        self.tokens_after_think_end = -1

    def _advance_state(self, token: int) -> None:
        if self.tokens_after_think_end == -1 and token == self.think_end_id:
            self.tokens_after_think_end = 0
        elif self.tokens_after_think_end >= 0:
            self.tokens_after_think_end += 1

    def _rollback_state(self) -> None:
        if self.tokens_after_think_end == 0:
            self.tokens_after_think_end = -1
        elif self.tokens_after_think_end > 0:
            self.tokens_after_think_end -= 1

    def accept_token(self, token: int) -> None:
        self.current_token = token
        if self.tokens_after_think_end >= 0:
            self.grammar.accept_token(token)
        self._advance_state(token)

    def rollback(self, k: int) -> None:
        steps_after_think = min(k, self.tokens_after_think_end)
        if steps_after_think > 0:
            self.grammar.rollback(steps_after_think)
        for _ in range(k):
            self._rollback_state()
        self.current_token = None

    def is_terminated(self) -> bool:
        return self.grammar.is_terminated()

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        return self.grammar.allocate_vocab_mask(vocab_size, batch_size, device)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self.tokens_after_think_end >= 0:
            self.grammar.fill_vocab_mask(vocab_mask, idx)

    def move_vocab_mask(self, vocab_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self.grammar.move_vocab_mask(vocab_mask, device)

    def apply_vocab_mask(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        self.grammar.apply_vocab_mask(logits, vocab_mask)

    def copy(self) -> BaseGrammarObject:
        return ReasonerGrammarObject(self.grammar.copy(), self.think_end_id)

    @property
    def finished(self) -> bool:
        return self.grammar.finished

    @finished.setter
    def finished(self, finished: bool) -> None:
        self.grammar.finished = finished


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(self, grammar_backend: BaseGrammarBackend, think_end_id: int) -> None:
        super().__init__()
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id

    def dispatch_json(self, key_string: str) -> BaseGrammarObject | None:
        ret = self.grammar_backend.dispatch_json(key_string)
        if ret is None or ret is INVALID_GRAMMAR_OBJ:
            return ret
        return ReasonerGrammarObject(ret, self.think_end_id)

    def reset(self) -> None:
        super().reset()
        self.grammar_backend.reset()

    def shutdown(self) -> None:
        super().shutdown()
        self.grammar_backend.shutdown()

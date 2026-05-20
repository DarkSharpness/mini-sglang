from __future__ import annotations

import json

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    allocate_token_bitmask,
)

from .base import INVALID_GRAMMAR_OBJ, BaseGrammarBackend, BaseGrammarObject

MAX_ROLLBACK_TOKENS = 200


def _apply_vocab_mask_torch(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    # NOTE: logits shape: [batch, vocab], vocab_mask shape: [batch, ceil(vocab / 32)].
    vocab_size = min(logits.shape[1], vocab_mask.shape[1] * 32)
    shifts = torch.arange(32, dtype=torch.int32, device=vocab_mask.device)
    unpacked = ((vocab_mask.unsqueeze(-1) >> shifts) & 1).reshape(vocab_mask.shape[0], -1)
    logits[:, :vocab_size].masked_fill_(~unpacked[:, :vocab_size].to(torch.bool), float("-inf"))


class XGrammarGrammar(BaseGrammarObject):
    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: list[int] | int | None,
        key_string: str | None = None,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.key_string = key_string
        self.accepted_tokens: list[int] = []

    def accept_token(self, token: int) -> None:
        if self.is_terminated():
            return
        self.current_token = token
        if not self.matcher.accept_token(token):
            raise ValueError(
                f"Token {token} not accepted by grammar {self.key_string!r}: "
                f"{self.accepted_tokens=}"
            )
        self.accepted_tokens.append(token)

    def rollback(self, k: int) -> None:
        self.matcher.rollback(k)
        if k > 0:
            self.accepted_tokens = self.accepted_tokens[:-k]
            self.current_token = self.accepted_tokens[-1] if self.accepted_tokens else None

    def is_terminated(self) -> bool:
        return self.matcher.is_terminated()

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        _apply_vocab_mask_torch(logits, vocab_mask)

    def copy(self) -> BaseGrammarObject:
        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
        )


class TokenizerNotSupportedError(Exception):
    pass


class XGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        eos_token_ids: int | list[int] | set[int] | None = None,
        any_whitespace: bool = True,
    ) -> None:
        super().__init__()
        try:
            tokenizer_info = TokenizerInfo.from_huggingface(
                tokenizer,
                vocab_size=vocab_size,
                stop_token_ids=self._normalize_eos_token_ids(eos_token_ids),
            )
        except Exception as e:
            raise TokenizerNotSupportedError(
                f"Failed to create XGrammar TokenizerInfo from tokenizer: {e}"
            ) from e
        override_stop_tokens = None

        self.vocab_size = vocab_size
        self.any_whitespace = any_whitespace
        self.override_stop_tokens = override_stop_tokens
        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)

    @staticmethod
    def _normalize_eos_token_ids(
        eos_token_ids: int | list[int] | set[int] | None,
    ) -> list[int] | None:
        if eos_token_ids is None:
            return None
        if isinstance(eos_token_ids, int):
            return [eos_token_ids]
        return list(eos_token_ids)

    def _from_context(self, ctx: CompiledGrammar, key_string: str) -> XGrammarGrammar:
        matcher = GrammarMatcher(
            ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            ctx,
            self.override_stop_tokens,
            key_string,
        )

    def dispatch_json(self, key_string: str) -> BaseGrammarObject | None:
        try:
            if key_string == "$$ANY$$":
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                ctx = self.grammar_compiler.compile_json_schema(
                    schema=key_string,
                    any_whitespace=self.any_whitespace,
                )
        except (RuntimeError, json.decoder.JSONDecodeError, UnicodeDecodeError):
            return INVALID_GRAMMAR_OBJ
        return self._from_context(ctx, key_string)

    def reset(self) -> None:
        super().reset()
        self.grammar_compiler.clear_cache()

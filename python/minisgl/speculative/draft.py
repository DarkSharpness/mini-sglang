"""Draft strategies for speculative decoding.

A draft instance must provide three methods (no `Protocol` declared in v1 — the
contract is enforced by duck typing; extract a protocol when the third strategy
arrives):

- ``warm_up(prompt_ids)``: feed the prompt so the draft's internal state mirrors
  the target's after its initial prefill. May be a no-op.
- ``draft(last_token, k)``: autoregressively generate ``k`` candidate token IDs
  starting from ``last_token``. Returns a Python list of ``k`` ints.
- ``rollback(rejected_count)``: discard internal state corresponding to the
  ``rejected_count`` rejected drafts. May be a no-op.
"""
from __future__ import annotations


class DummyDraft:
    """Always returns the same wrong token. Used to exercise the rejection path."""

    def __init__(self, k: int, wrong_token: int = 100_000) -> None:
        self.k = k
        self.wrong_token = wrong_token

    def warm_up(self, prompt_ids: list[int]) -> None:
        pass

    def draft(self, last_token: int, k: int) -> list[int]:
        assert k == self.k
        return [self.wrong_token] * k

    def rollback(self, rejected_count: int) -> None:
        pass

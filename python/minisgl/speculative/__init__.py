"""Vanilla speculative decoding: single-request, greedy verification only."""
from .draft import DummyDraft
from .engine import SpeculativeEngine
from .verify import verify_drafts

__all__ = ["DummyDraft", "SpeculativeEngine", "verify_drafts"]

"""Vanilla speculative decoding: single-request, greedy verification only."""
from .draft import DummyDraft, StandaloneDraft
from .engine import SpeculativeEngine
from .verify import verify_drafts

__all__ = ["DummyDraft", "SpeculativeEngine", "StandaloneDraft", "verify_drafts"]

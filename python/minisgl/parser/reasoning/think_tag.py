from __future__ import annotations

from .base import BaseDetector


class ThinkTagDetector(BaseDetector):
    """
    Detector for the ``<think>…</think>`` reasoning format.

    Used by DeepSeek-R1, Qwen3-Thinking, QwQ, and other models that wrap
    chain-of-thought content in ``<think>`` / ``</think>`` XML-like tags.
    """

    @property
    def start_tag(self) -> str:
        return "<think>"

    @property
    def end_tag(self) -> str:
        return "</think>"

"""
minisgl.parser.reasoning
========================

Utilities for identifying and splitting reasoning blocks from model-generated
text.  Supports both **non-streaming** (full-text) and **streaming**
(chunk-by-chunk) parsing.

Quick start
-----------
::

    from minisgl.parser.reasoning import ReasoningParser

    # Non-streaming
    parser = ReasoningParser("qwen3")
    result = parser.parse_full(full_text)
    print(result.reasoning_text)  # chain-of-thought
    print(result.normal_text)     # answer

    # Streaming
    parser = ReasoningParser("qwen3")
    for raw_chunk in token_stream:
        sc = parser.parse_stream(raw_chunk)
        ...
    final = parser.flush()

Public API
----------
:class:`ReasoningParser`
    Main entry point.  Selects a detector by *model_type* and exposes
    :meth:`~ReasoningParser.parse_full`, :meth:`~ReasoningParser.parse_stream`,
    :meth:`~ReasoningParser.flush`, and :meth:`~ReasoningParser.stream_iter`.

:class:`ParseResult`
    Non-streaming result with ``reasoning_text`` and ``normal_text``.

:class:`StreamChunk`
    Streaming result with ``reasoning_delta`` and ``normal_delta``.

:class:`BaseDetector`
    ABC for custom detectors (subclass to add new reasoning formats).

:class:`ThinkTagDetector`
    Concrete detector for the ``<think>…</think>`` format used by
    Qwen3-Thinking.
"""

from .base import BaseDetector, ParseResult, StreamChunk
from .parser import ReasoningParser
from .think_tag import ThinkTagDetector

__all__ = [
    "ReasoningParser",
    "ParseResult",
    "StreamChunk",
    "BaseDetector",
    "ThinkTagDetector",
]

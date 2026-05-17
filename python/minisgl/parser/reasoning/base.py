from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Non-streaming parse output."""

    reasoning_text: str
    """Content extracted from inside the reasoning block."""

    normal_text: str
    """Remaining content outside the reasoning block."""


@dataclass
class StreamChunk:
    """Incremental output produced by a single :meth:`ReasoningParser.parse_stream` call."""

    reasoning_delta: str
    """New reasoning content decoded in this chunk (may be empty)."""

    normal_delta: str
    """New normal content decoded in this chunk (may be empty)."""


class BaseDetector(ABC):
    """
    Abstract base for reasoning-block detectors.

    A detector knows the start/end markers that delimit a reasoning block
    and can parse both complete text (non-streaming) and incremental chunks
    (streaming) through the helpers in :class:`ReasoningParser`.

    Concrete subclasses must implement :attr:`start_tag` and :attr:`end_tag`.
    They may also override :meth:`parse_full` for model-specific logic.
    """

    @property
    @abstractmethod
    def start_tag(self) -> str:
        """Opening delimiter of the reasoning block (e.g. ``"<think>"```)."""

    @property
    @abstractmethod
    def end_tag(self) -> str:
        """Closing delimiter of the reasoning block (e.g. ``"</think>"```)."""

    def parse_full(self, text: str) -> ParseResult:
        """
        Extract reasoning and normal text from a complete, fully-decoded
        generation string (non-streaming path).

        The default implementation does a single forward scan for the
        first occurrence of :attr:`start_tag` / :attr:`end_tag`.  Subclasses
        may override for more sophisticated (e.g. multi-block) handling.
        """
        start = self.start_tag
        end = self.end_tag

        start_idx = text.find(start)
        if start_idx == -1:
            return ParseResult(reasoning_text="", normal_text=text)

        content_start = start_idx + len(start)
        end_idx = text.find(end, content_start)

        if end_idx == -1:
            # Reasoning block was never closed – treat everything after the
            # start tag as reasoning content and keep the preceding text as
            # normal output.
            return ParseResult(
                reasoning_text=text[content_start:],
                normal_text=text[:start_idx],
            )

        reasoning = text[content_start:end_idx]
        normal = text[:start_idx] + text[end_idx + len(end):]
        return ParseResult(reasoning_text=reasoning, normal_text=normal)

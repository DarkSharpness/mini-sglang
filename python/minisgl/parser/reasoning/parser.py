from __future__ import annotations

from typing import Iterator

from .base import BaseDetector, ParseResult, StreamChunk
from .think_tag import ThinkTagDetector

# ---------------------------------------------------------------------------
# Model-type → detector mapping
# ---------------------------------------------------------------------------

# Model types (lowercased, hyphens replaced by underscores) that are known
# to use the <think>…</think> format.
_THINK_TAG_MODELS: frozenset[str] = frozenset(
    {
        "qwen3",
        "qwen3_moe",
    }
)

# Prefixes that are also mapped to ThinkTagDetector even when not in the
# exact-match set above (handles future model variants automatically).
_THINK_TAG_PREFIXES: tuple[str, ...] = ("qwen3",)


# ---------------------------------------------------------------------------
# Streaming state machine
# ---------------------------------------------------------------------------


def _safe_prefix_len(buf: str, tag: str) -> int:
    """
    Number of leading characters in *buf* that are guaranteed **not** to be
    the beginning of an upcoming *tag*.

    We keep at most ``len(tag) - 1`` trailing characters buffered so that a
    tag that is split across two successive chunks is never emitted early.

    Examples::

        _safe_prefix_len("hello <thi",  "<think>")  → 4   # "hell" is safe
        _safe_prefix_len("hello",       "<think>")  → 0   # all could match
        _safe_prefix_len("hello world", "<think>")  → 5
    """
    return max(0, len(buf) - len(tag) + 1)


class _StreamingState:
    """
    Internal stateful helper that processes incremental text chunks and
    produces per-chunk ``(reasoning_delta, normal_delta)`` pairs.

    State machine
    -------------
    ``before``
        The reasoning block has not started yet.  Incoming text is treated as
        normal output until the start tag is found.

    ``in_reasoning``
        We are inside the reasoning block.  Incoming text is treated as
        reasoning content until the end tag is found.

    ``after``
        The reasoning block has closed.  All remaining text is normal output.

    Partial-tag handling
    --------------------
    Because chunks can arrive mid-tag (e.g. ``"<thi"`` + ``"nk>"``), we
    buffer the last ``len(tag) - 1`` characters in each state and only emit
    characters that are definitively before any possible tag boundary.
    """

    def __init__(self, detector: BaseDetector) -> None:
        self._detector = detector
        self._state: str = "before"
        self._buf: str = ""

    def feed(self, chunk: str) -> StreamChunk:
        """
        Consume one text chunk and return the incremental reasoning / normal
        content decoded so far.

        This method is **not** re-entrant; call it sequentially for each
        streamed chunk and call :meth:`flush` once after the final chunk.
        """
        reasoning_delta = ""
        normal_delta = ""

        self._buf += chunk

        # ------------------------------------------------------------------ #
        # Phase 1 – scan for the reasoning block start tag                   #
        # ------------------------------------------------------------------ #
        if self._state == "before":
            start_tag = self._detector.start_tag
            idx = self._buf.find(start_tag)
            if idx != -1:
                # Everything before the tag is normal text.
                normal_delta += self._buf[:idx]
                # Discard the tag itself and advance.
                self._buf = self._buf[idx + len(start_tag):]
                self._state = "in_reasoning"
            else:
                # Emit the safe prefix; keep a potential partial-tag suffix.
                safe = _safe_prefix_len(self._buf, start_tag)
                normal_delta += self._buf[:safe]
                self._buf = self._buf[safe:]

        # ------------------------------------------------------------------ #
        # Phase 2 – scan for the reasoning block end tag                     #
        # (runs in the same call if the start tag was just found above)      #
        # ------------------------------------------------------------------ #
        if self._state == "in_reasoning":
            end_tag = self._detector.end_tag
            idx = self._buf.find(end_tag)
            if idx != -1:
                reasoning_delta += self._buf[:idx]
                self._buf = self._buf[idx + len(end_tag):]
                self._state = "after"
            else:
                safe = _safe_prefix_len(self._buf, end_tag)
                reasoning_delta += self._buf[:safe]
                self._buf = self._buf[safe:]

        # ------------------------------------------------------------------ #
        # Phase 3 – emit everything remaining as normal text                 #
        # ------------------------------------------------------------------ #
        if self._state == "after":
            normal_delta += self._buf
            self._buf = ""

        return StreamChunk(reasoning_delta=reasoning_delta, normal_delta=normal_delta)

    def flush(self) -> StreamChunk:
        """
        Signal end-of-stream.

        Drains any text that was kept in the internal buffer to guard against
        partial tags.  Should be called **once** after the final chunk so that
        callers never miss trailing content.
        """
        remaining = self._buf
        self._buf = ""
        if self._state == "in_reasoning":
            # Unclosed reasoning block – treat remainder as reasoning.
            return StreamChunk(reasoning_delta=remaining, normal_delta="")
        return StreamChunk(reasoning_delta="", normal_delta=remaining)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ReasoningParser:
    """
    High-level reasoning-content parser.

    Selects the appropriate detector based on *model_type* and exposes a
    unified interface for both **non-streaming** (single-call) and
    **streaming** (chunk-by-chunk) use.

    Parameters
    ----------
    model_type:
        The ``model_type`` string from the HuggingFace ``config.json``
        (e.g. ``"qwen3"``).  Pass ``None`` or ``"auto"``
        to use :class:`ThinkTagDetector` as a sensible default.

    Examples
    --------
    Non-streaming::

        parser = ReasoningParser("qwen3")
        result = parser.parse_full(full_generated_text)
        print(result.reasoning_text)   # chain-of-thought
        print(result.normal_text)      # answer

    Streaming::

        parser = ReasoningParser("qwen3")
        for raw_chunk in token_stream:
            chunk = parser.parse_stream(raw_chunk)
            if chunk.reasoning_delta:
                handle_reasoning(chunk.reasoning_delta)
            if chunk.normal_delta:
                handle_normal(chunk.normal_delta)
        # drain remaining buffer
        final = parser.flush()
        if final.normal_delta:
            handle_normal(final.normal_delta)
    """

    def __init__(self, model_type: str | None = None) -> None:
        self._detector: BaseDetector = self._make_detector(model_type)
        self._streaming_state = _StreamingState(self._detector)

    # ---------------------------------------------------------------------- #
    # Non-streaming                                                           #
    # ---------------------------------------------------------------------- #

    def parse_full(self, text: str) -> ParseResult:
        """
        Parse a **complete** generation string.

        Returns a :class:`ParseResult` with ``reasoning_text`` (content
        inside the reasoning block) and ``normal_text`` (everything else).
        """
        return self._detector.parse_full(text)

    # ---------------------------------------------------------------------- #
    # Streaming                                                               #
    # ---------------------------------------------------------------------- #

    def parse_stream(self, chunk: str) -> StreamChunk:
        """
        Feed one incremental text chunk from a streaming decode loop.

        Returns a :class:`StreamChunk` whose ``reasoning_delta`` and
        ``normal_delta`` fields hold the content that can be safely attributed
        to reasoning and normal output respectively for this chunk.

        Call :meth:`flush` after the **last** chunk to drain the internal
        partial-tag buffer.
        """
        return self._streaming_state.feed(chunk)

    def flush(self) -> StreamChunk:
        """
        Signal end-of-stream and drain any buffered content.

        Must be called once after the final chunk to ensure no trailing text
        is silently dropped.
        """
        return self._streaming_state.flush()

    def stream_iter(self, chunks: Iterator[str]) -> Iterator[StreamChunk]:
        """
        Convenience wrapper: iterate over *chunks* and yield one
        :class:`StreamChunk` per input chunk, followed by the flush chunk.

        Usage::

            for sc in parser.stream_iter(token_stream):
                ...
        """
        for chunk in chunks:
            yield self.parse_stream(chunk)
        final = self.flush()
        if final.reasoning_delta or final.normal_delta:
            yield final

    # ---------------------------------------------------------------------- #
    # Internals                                                               #
    # ---------------------------------------------------------------------- #

    @property
    def detector(self) -> BaseDetector:
        """The underlying :class:`BaseDetector` instance (read-only)."""
        return self._detector

    @staticmethod
    def _make_detector(model_type: str | None) -> BaseDetector:
        """
        Resolve the detector for the given *model_type*.

        Resolution order
        ----------------
        1. ``None`` / ``"auto"``  → :class:`ThinkTagDetector` (safe default)
        2. Exact match in ``_THINK_TAG_MODELS`` set
        3. Prefix match against ``_THINK_TAG_PREFIXES``
        4. Unrecognised model type → :class:`ThinkTagDetector` (safe fallback)
           The caller is responsible for knowing whether the model actually
           emits reasoning tags; an unknown type simply won't extract any
           reasoning if no ``<think>`` tag is present.
        """
        if model_type is None or model_type == "auto":
            return ThinkTagDetector()

        normalized = model_type.lower().replace("-", "_")

        if normalized in _THINK_TAG_MODELS:
            return ThinkTagDetector()

        if any(normalized.startswith(p) for p in _THINK_TAG_PREFIXES):
            return ThinkTagDetector()

        # Unknown model type – fall back gracefully.
        return ThinkTagDetector()

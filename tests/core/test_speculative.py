from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.env import ENV
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.scheduler import _validate_spec_config
from minisgl.scheduler.spec import accept_greedy, propose_ngram, resolve_verify


# Drafter: suffix matching must be deterministic, bounded, and robust to tensor layout.
@pytest.mark.parametrize(
    ("history", "min_n", "max_n", "budget", "expected"),
    [
        ([1, 2, 7, 1, 2, 8, 1, 2], 2, 3, 3, [8, 1, 2]),  # most recent match
        ([1, 2, 9, 2, 8, 1, 2], 1, 2, 1, [9]),  # longest before more-recent
        ([4, 9, 6, 4, 7, 4], 1, 3, 2, [7, 4]),  # fallback to shorter n-gram
        ([1, 2, 3, 4], 2, 3, 2, []),  # no suffix match
        ([1], 1, 3, 4, []),  # history too short
        ([1, 1], 1, 3, 0, []),  # no draft budget
    ],
    ids=["recent", "longest", "shorter-fallback", "no-match", "short-history", "zero-budget"],
)
def test_propose_ngram(
    history: list[int], min_n: int, max_n: int, budget: int, expected: list[int]
) -> None:
    """Choose the longest match, break ties by recency, and respect the draft budget."""
    tokens = torch.tensor(history, dtype=torch.int32)
    assert propose_ngram(
        tokens, min_ngram=min_n, max_ngram=max_n, max_tokens=budget
    ) == expected


# Acceptance: keep the matching draft prefix and always add one target-model token.
@pytest.mark.parametrize(
    ("drafts", "targets", "expected"),
    [
        ([1, 2], [9, 8, 7], ([9], 0)),
        ([11, 22, 33], [11, 99, 44, 55], ([11, 99], 1)),
        ([1, 2], [1, 2, 3], ([1, 2, 3], 2)),
    ],
    ids=["reject-first", "partial", "accept-all-plus-bonus"],
)
def test_accept_greedy(
    drafts: list[int], targets: list[int], expected: tuple[list[int], int]
) -> None:
    """Reject-first, partial-accept, and all-accepted-plus-bonus behavior."""
    assert accept_greedy(drafts, targets) == expected


@pytest.mark.parametrize(
    ("targets", "remain", "eos", "ignore_eos", "expected"),
    [
        ([1, 2, 3], 8, 2, False, ([1, 2], 2, True)),
        ([1, 2, 3], 1, 99, False, ([1], 2, True)),
        ([9, 8, 7], 8, None, True, ([9], 0, False)),
    ],
    ids=["eos", "max-tokens-keeps-raw-accept-count", "correction"],
)
def test_resolve_verify(
    targets: list[int],
    remain: int,
    eos: int | None,
    ignore_eos: bool,
    expected: tuple[list[int], int, bool],
) -> None:
    """Apply EOS/max-token truncation after acceptance without changing raw metrics."""
    assert resolve_verify(
        [1, 2],
        targets,
        remain_len=remain,
        eos_token_id=eos,
        ignore_eos=ignore_eos,
    ) == expected


def test_request_forward_length_keeps_committed_state_separate() -> None:
    """Temporary drafts extend the forward only; committed length/remain_len stay honest."""
    req = _req(
        [10, 20, 30],
        table_idx=0,
        cached_len=2,
        draft_tokens=[40, 50],
    )

    assert (req.cached_len, req.device_len, req.forward_device_len) == (2, 3, 5)
    assert (req.extend_len, req.remain_len) == (3, 8)


def test_decode_manager_separates_greedy_and_sampled_requests() -> None:
    """Only greedy requests enter verify; sampled requests retain normal decoding."""
    greedy = _req([1, 2, 7, 1, 2], table_idx=0, cached_len=4)
    sampled = _req([3, 4], table_idx=1, cached_len=1, temperature=0.7)
    manager = DecodeManager(
        page_size=1,
        spec_algorithm="ngram",
        spec_num_draft=2,
        spec_ngram_min=2,
        spec_ngram_max=2,
        running_reqs={greedy, sampled},
    )

    verify_batch = manager.schedule_next_batch()
    sampled_batch = manager.schedule_next_batch()

    assert verify_batch is not None and verify_batch.is_verify
    assert verify_batch.reqs == [greedy]
    assert greedy.draft_tokens == [7, 1]
    assert sampled_batch is not None and sampled_batch.is_decode
    assert sampled_batch.reqs == [sampled]


def test_decode_manager_caps_drafts_to_leave_bonus_slot() -> None:
    """Never draft into the final output slot: verification needs room for one target token."""
    req = _req([1, 2, 7, 1, 2], table_idx=0, cached_len=4, output_len=2)
    manager = _manager(req, spec_num_draft=4)

    batch = manager.schedule_next_batch()

    assert batch is not None and batch.is_verify
    assert req.draft_tokens == [7]


def test_decode_manager_falls_back_when_no_request_drafts() -> None:
    """An all-miss step uses normal decode, preserving its CUDA-graph fast path."""
    reqs = {
        _req([1, 2, 3], table_idx=0, cached_len=2),
        _req([4, 5, 6], table_idx=1, cached_len=2),
    }
    batch = _manager(*reqs).schedule_next_batch()

    assert batch is not None and batch.is_decode
    assert all(not req.draft_tokens for req in reqs)


def test_decode_manager_supports_ragged_verify_drafts() -> None:
    """One drafting request can share verify with a sibling that proposed zero tokens."""
    matched = _req([1, 2, 7, 1, 2], table_idx=0, cached_len=4)
    missed = _req([4, 5, 6], table_idx=1, cached_len=2)

    batch = _manager(matched, missed).schedule_next_batch()

    assert batch is not None and batch.is_verify
    assert [len(req.draft_tokens) for req in batch.reqs] == [2, 0]


def test_rollback_paged_batch_returns_exact_physical_slots() -> None:
    """Rollback frees only rejected physical pages; empty ranges are harmless."""
    manager = CacheManager.__new__(CacheManager)
    manager.page_size = 1
    manager.page_table = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    manager.free_slots = torch.tensor([99])

    manager.rollback_paged_batch([(0, 1, 3), (1, 2, 2), (1, 0, 1)])

    assert manager.free_slots.tolist() == [99, 11, 12, 20]


def _valid_ngram_config(**overrides: object) -> SchedulerConfig:
    base = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        page_size=1,
        spec_algorithm="ngram",
        spec_num_draft=4,
        spec_ngram_min=1,
        spec_ngram_max=3,
    )
    return replace(base, **overrides) if overrides else base


@pytest.mark.parametrize("attention_backend", ["fi", "fa"])
def test_spec_config_accepts_supported_ngram_settings(
    attention_backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validated spec surface is TP=1, page_size=1, overlap-off, and FI or FA."""
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    _validate_spec_config(_valid_ngram_config(attention_backend=attention_backend))


def test_spec_config_skips_gates_when_speculation_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec-off must stay bring-up compatible with fa / large pages / TP / overlap."""
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    _validate_spec_config(
        _valid_ngram_config(
            spec_algorithm="none",
            attention_backend="fa",
            page_size=16,
            tp_info=DistributedInfo(0, 2),
        )
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"spec_algorithm": "eagle"}, "Unsupported speculative algorithm"),
        ({"tp_info": DistributedInfo(0, 2)}, "TP=1"),
        ({"page_size": 16}, "page_size=1"),
        ({"attention_backend": "fa,fi"}, "attention-backend fi or fa"),
        ({"attention_backend": "auto"}, "attention-backend fi or fa"),
        ({"spec_num_draft": 0}, "spec_num_draft must be at least 1"),
        ({"spec_ngram_min": 0, "spec_ngram_max": 3}, "1 <= spec_ngram_min <= spec_ngram_max"),
        ({"spec_ngram_min": 4, "spec_ngram_max": 3}, "1 <= spec_ngram_min <= spec_ngram_max"),
    ],
    ids=[
        "bad-algorithm",
        "tp-gt-1",
        "page-size-gt-1",
        "attn-hybrid-fa-fi",
        "attn-auto",
        "draft-budget-zero",
        "ngram-min-too-small",
        "ngram-min-gt-max",
    ],
)
def test_spec_config_errors_clearly_on_unsupported_settings(
    overrides: dict[str, object], match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every unsupported knob fails at startup with an actionable message."""
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    with pytest.raises(ValueError, match=match):
        _validate_spec_config(_valid_ngram_config(**overrides))


def test_spec_config_requires_overlap_scheduling_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speculation rejects overlap scheduling because variable commits are not supported there."""
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    with pytest.raises(ValueError, match="DISABLE_OVERLAP"):
        _validate_spec_config(_valid_ngram_config())


def _manager(*reqs: Req, spec_num_draft: int = 2) -> DecodeManager:
    return DecodeManager(
        page_size=1,
        spec_algorithm="ngram",
        spec_num_draft=spec_num_draft,
        spec_ngram_min=2,
        spec_ngram_max=2,
        running_reqs=set(reqs),
    )


def _req(
    tokens: list[int],
    *,
    table_idx: int,
    cached_len: int,
    output_len: int = 8,
    temperature: float = 0.0,
    draft_tokens: list[int] | None = None,
) -> Req:
    return Req(
        input_ids=torch.tensor(tokens, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=output_len,
        uid=table_idx,
        sampling_params=SamplingParams(temperature=temperature),
        cache_handle=None,  # type: ignore[arg-type]
        draft_tokens=draft_tokens or [],
    )

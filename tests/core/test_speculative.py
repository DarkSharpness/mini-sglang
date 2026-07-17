from __future__ import annotations

import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.spec import accept_greedy, propose_ngram, resolve_verify


@pytest.mark.parametrize(
    ("history", "min_n", "max_n", "budget", "expected"),
    [
        ([1, 2, 7, 1, 2, 8, 1, 2], 2, 3, 3, [8, 1, 2]),  # most recent match
        ([1, 2, 9, 2, 8, 1, 2], 1, 2, 1, [9]),  # longest before more-recent
        ([4, 9, 6, 4, 7, 4], 1, 3, 2, [7, 4]),  # fallback to shorter n-gram
        ([1], 1, 3, 4, []),  # history too short
        ([1, 1], 1, 3, 0, []),  # no draft budget
    ],
    ids=["recent", "longest", "shorter-fallback", "short-history", "zero-budget"],
)
def test_propose_ngram(
    history: list[int], min_n: int, max_n: int, budget: int, expected: list[int]
) -> None:
    tokens = torch.tensor(history, dtype=torch.int32)
    assert propose_ngram(
        tokens, min_ngram=min_n, max_ngram=max_n, max_tokens=budget
    ) == expected


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
    assert resolve_verify(
        [1, 2],
        targets,
        remain_len=remain,
        eos_token_id=eos,
        ignore_eos=ignore_eos,
    ) == expected


def test_request_forward_length_keeps_committed_state_separate() -> None:
    req = _req(
        [10, 20, 30],
        table_idx=0,
        cached_len=2,
        draft_tokens=[40, 50],
    )

    assert (req.cached_len, req.device_len, req.forward_device_len) == (2, 3, 5)
    assert (req.extend_len, req.remain_len) == (3, 8)


def test_decode_manager_separates_greedy_and_sampled_requests() -> None:
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


def _req(
    tokens: list[int],
    *,
    table_idx: int,
    cached_len: int,
    temperature: float = 0.0,
    draft_tokens: list[int] | None = None,
) -> Req:
    return Req(
        input_ids=torch.tensor(tokens, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=8,
        uid=table_idx,
        sampling_params=SamplingParams(temperature=temperature),
        cache_handle=None,  # type: ignore[arg-type]
        draft_tokens=draft_tokens or [],
    )

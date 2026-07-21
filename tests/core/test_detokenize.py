from __future__ import annotations

from typing import Dict, List

from minisgl.message import DetokenizeMsg
from minisgl.tokenizer.detokenize import DetokenizeManager

# BPE-style vocab: " Lorraine" splits mid-word, so any re-emission of an
# already-sent token is visible as overlapping text ("Lorraineorraine ...").
VOCAB = {0: "<eos>", 1: " L", 2: "orraine", 3: " starts", 4: " with", 5: " Hello"}


class _StubTokenizer:
    eos_token_id = 0

    def batch_decode(self, id_lists: List[List[int]]) -> List[str]:
        return ["".join(VOCAB[i] for i in ids) for ids in id_lists]


def _stream(batches: List[List[DetokenizeMsg]]) -> Dict[int, str]:
    """Feed batches through one manager and concatenate per-uid increments."""
    manager = DetokenizeManager(_StubTokenizer())  # type: ignore[arg-type]
    texts: Dict[int, str] = {}
    for batch in batches:
        for msg, incremental in zip(batch, manager.detokenize(batch), strict=True):
            texts[msg.uid] = texts.get(msg.uid, "") + incremental
    return texts


# A verify step commits up to K+1 tokens as one DetokenizeMsg per token, and the
# tokenizer server drains them into a single detokenize() batch.
def test_multi_commit_batch_matches_one_per_step() -> None:
    """Same-uid msgs in one batch must stream the same text as one-msg batches."""
    tokens = (1, 2, 3, 4)  # " L" + "orraine" + " starts" + " with"
    # Spec-off delivery — [[m1], [m2], [m3], [m4]]: four detokenize() calls,
    # offsets update between tokens. Defines the expected text.
    one_per_step = _stream([[DetokenizeMsg(uid=7, next_token=t, finished=False)] for t in tokens])
    # Verify-step delivery — [[m1, m2, m3, m4]]: K accepted drafts + bonus
    # emitted as one msg per token, drained into a single detokenize() call.
    multi_commit = _stream([[DetokenizeMsg(uid=7, next_token=t, finished=False) for t in tokens]])
    assert one_per_step[7] == " Lorraine starts with"
    # Pre-fix, stale offsets re-emitted sent text here:
    # " L" + " Lorraine" + " Lorraine starts" + " Lorraine starts with"
    assert multi_commit[7] == one_per_step[7]


def test_interleaved_uids_stay_isolated() -> None:
    """Multi-commit batches for several uids must not leak text across uids."""
    # One call, two uids interleaved — [[u1m1, u2m1, u1m2, u2m2]]; each uid emits once.
    texts = _stream(
        [
            [
                DetokenizeMsg(uid=1, next_token=1, finished=False),  # uid1: " L"
                DetokenizeMsg(uid=2, next_token=5, finished=False),  # uid2: " Hello"
                DetokenizeMsg(uid=1, next_token=2, finished=False),  # uid1: "orraine"
                DetokenizeMsg(uid=2, next_token=3, finished=False),  # uid2: " starts"
            ]
        ]
    )
    assert texts == {1: " Lorraine", 2: " Hello starts"}


def test_finish_inside_multi_commit_batch() -> None:
    """A finishing EOS in the same batch flushes prior tokens exactly once."""
    # One call — [[m1, m2=EOS+finished]]; flush once, then drop the uid's state.
    texts = _stream(
        [
            [
                DetokenizeMsg(uid=7, next_token=3, finished=False),  # " starts"
                DetokenizeMsg(uid=7, next_token=0, finished=True),  # EOS: never emitted
            ]
        ]
    )
    assert texts[7] == " starts"

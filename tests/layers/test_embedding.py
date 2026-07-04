"""Gate 1.9l: hermetic tests for VocabParallelEmbedding CUDA/NPU/CPU dispatch.

Verifies:
  * import + construction do not touch minisgl.kernel
  * CPU/NPU TP=1 uses F.embedding directly, no dtype coercion
  * CPU/NPU TP>1 masked-gather + zero-fill + all_reduce
  * CUDA branch still calls the lazy minisgl.kernel.indexing with
    vocab_range=(start,length) for TP>1
  * ParallelLMHead tied path is unaffected
  * source does not contain .long() / .to(torch.int64)
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
EMB_SRC = REPO / "python" / "minisgl" / "layers" / "embedding.py"


# ------------------------------------------------------------------ fixtures
def _reset_tp_info():
    """Reset the module-global TP info singleton so tests can set it fresh."""
    import minisgl.distributed.info as info_mod
    info_mod._TP_INFO = None


def _set_tp(rank, size):
    _reset_tp_info()
    from minisgl.distributed import set_tp_info
    set_tp_info(rank=rank, size=size)


@pytest.fixture(autouse=True)
def _tp_reset():
    _reset_tp_info()
    yield
    _reset_tp_info()


@pytest.fixture()
def _tp1():
    _set_tp(0, 1)
    yield


@pytest.fixture()
def _clean_kernel(monkeypatch):
    """Remove any stale minisgl.kernel entries so we can observe fresh imports."""
    for m in list(sys.modules):
        if m == "minisgl.kernel" or m.startswith("minisgl.kernel."):
            monkeypatch.delitem(sys.modules, m, raising=False)
    yield


class _FakeComm:
    """Fake DistributedCommunicator recording all_reduce/all_gather invocations."""

    def __init__(self):
        self.all_reduce_calls = 0
        self.all_gather_calls = 0

    def all_reduce(self, x):
        self.all_reduce_calls += 1
        # In tests we simulate the sum by returning x unchanged;
        # dedicated TP>1 tests do the manual per-rank sum.
        return x

    def all_gather(self, x):
        self.all_gather_calls += 1
        return x


def _make_emb(vocab, hidden, *, tp_rank=0, tp_size=1, weight=None, seed=1234):
    """Construct a fresh VocabParallelEmbedding with a real weight tensor."""
    from minisgl.layers.embedding import VocabParallelEmbedding
    _set_tp(tp_rank, tp_size)
    emb = VocabParallelEmbedding(num_embeddings=vocab, embedding_dim=hidden)
    if weight is None:
        g = torch.Generator().manual_seed(seed)
        weight = torch.randn(emb.num_embeddings_tp, hidden, generator=g)
    else:
        assert tuple(weight.shape) == (emb.num_embeddings_tp, hidden)
    emb.weight = torch.nn.Parameter(weight)
    return emb


# ================================================================ 1. import
def test_import_does_not_load_minisgl_kernel(_clean_kernel):
    """Fresh import of the module must not pull minisgl.kernel."""
    sys.modules.pop("minisgl.layers.embedding", None)
    importlib.import_module("minisgl.layers.embedding")
    hits = [m for m in sys.modules if m == "minisgl.kernel" or m.startswith("minisgl.kernel.")]
    assert hits == [], f"unexpected kernel imports on module load: {hits}"


def test_construction_does_not_load_minisgl_kernel(_clean_kernel, _tp1):
    from minisgl.layers.embedding import VocabParallelEmbedding
    for m in list(sys.modules):
        if m == "minisgl.kernel" or m.startswith("minisgl.kernel."):
            sys.modules.pop(m, None)
    _ = VocabParallelEmbedding(num_embeddings=32, embedding_dim=16)
    hits = [m for m in sys.modules if m == "minisgl.kernel" or m.startswith("minisgl.kernel.")]
    assert hits == [], f"construction loaded kernel: {hits}"


# ================================================================ 2-6. CPU TP=1
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cpu_tp1_native_embedding(_tp1, dtype):
    torch.manual_seed(101)
    W = torch.randn(32, 16)
    emb = _make_emb(vocab=32, hidden=16, weight=W)
    ids = torch.tensor([3, 0, 17, 8, 31, 1, 15, 25, 12, 4], dtype=dtype)
    ids_before = ids.detach().clone()
    y = emb.forward(ids)
    assert tuple(y.shape) == (10, 16)
    assert y.dtype == W.dtype
    assert torch.equal(y, torch.nn.functional.embedding(ids, W))
    # inputs untouched, dtype preserved
    assert ids.dtype == dtype
    assert torch.equal(ids, ids_before)


def test_cpu_tp1_1d_shape(_tp1):
    W = torch.randn(32, 16)
    emb = _make_emb(vocab=32, hidden=16, weight=W)
    ids = torch.tensor([0, 5, 31], dtype=torch.int32)
    y = emb.forward(ids)
    assert tuple(y.shape) == (3, 16)


def test_cpu_tp1_2d_shape(_tp1):
    W = torch.randn(32, 16)
    emb = _make_emb(vocab=32, hidden=16, weight=W)
    ids = torch.tensor([[3, 0, 17, 8, 31], [1, 15, 25, 12, 4]], dtype=torch.int64)
    y = emb.forward(ids)
    assert tuple(y.shape) == (2, 5, 16)
    assert torch.equal(y, torch.nn.functional.embedding(ids, W))


def test_cpu_tp1_no_kernel_import(_tp1, _clean_kernel):
    W = torch.randn(32, 16)
    emb = _make_emb(vocab=32, hidden=16, weight=W)
    for m in list(sys.modules):
        if m == "minisgl.kernel" or m.startswith("minisgl.kernel."):
            sys.modules.pop(m, None)
    ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    emb.forward(ids)
    hits = [m for m in sys.modules if m == "minisgl.kernel" or m.startswith("minisgl.kernel.")]
    assert hits == [], f"NPU/CPU TP=1 forward loaded kernel: {hits}"


# ================================================================ 7-11. TP>1
def _tp2_setup(vocab=10, hidden=4, ids=None):
    torch.manual_seed(2002)
    W_full = torch.randn(vocab, hidden)
    if ids is None:
        ids = torch.tensor([0, 4, 5, 9], dtype=torch.int64)
    return W_full, ids


def _shard(W_full, tp_size, rank, num_embeddings_tp):
    start = num_embeddings_tp * rank
    finish = min(start + num_embeddings_tp, W_full.shape[0])
    W_r = torch.zeros(num_embeddings_tp, W_full.shape[1])
    length = finish - start
    W_r[:length] = W_full[start:finish]
    return W_r


def test_tp2_rank0_masked_gather(_clean_kernel):
    from minisgl.layers.embedding import VocabParallelEmbedding
    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    _set_tp(0, 2)
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator", new=_FakeComm
    ):
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb.weight = torch.nn.Parameter(_shard(W_full, 2, 0, emb.num_embeddings_tp))
    assert emb.vocab_range == (0, 5)
    y = emb.forward(ids)
    # rank 0 owns [0,5); ids [0,4,5,9] → only positions 0,1 valid
    assert tuple(y.shape) == (4, 4)
    assert torch.equal(y[0], W_full[0])
    assert torch.equal(y[1], W_full[4])
    assert torch.equal(y[2], torch.zeros(4))
    assert torch.equal(y[3], torch.zeros(4))
    assert emb._comm.all_reduce_calls == 1


def test_tp2_rank1_masked_gather(_clean_kernel):
    from minisgl.layers.embedding import VocabParallelEmbedding
    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    _set_tp(1, 2)
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator", new=_FakeComm
    ):
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb.weight = torch.nn.Parameter(_shard(W_full, 2, 1, emb.num_embeddings_tp))
    assert emb.vocab_range == (5, 5)
    y = emb.forward(ids)
    # rank 1 owns [5,10); only positions 2,3 valid
    assert torch.equal(y[0], torch.zeros(4))
    assert torch.equal(y[1], torch.zeros(4))
    assert torch.equal(y[2], W_full[5])
    assert torch.equal(y[3], W_full[9])
    assert emb._comm.all_reduce_calls == 1


def test_tp2_sum_matches_full_embedding(_clean_kernel):
    from minisgl.layers.embedding import VocabParallelEmbedding
    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    outs = []
    for r in range(2):
        _set_tp(r, 2)
        with mock.patch(
            "minisgl.layers.embedding.DistributedCommunicator", new=_FakeComm
        ):
            emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
        emb.weight = torch.nn.Parameter(_shard(W_full, 2, r, emb.num_embeddings_tp))
        outs.append(emb.forward(ids))
    total = outs[0] + outs[1]
    ref = W_full[ids]
    assert torch.equal(total, ref)


def test_tp2_vocab_range_start_length_contract(_clean_kernel):
    from minisgl.layers.embedding import VocabParallelEmbedding
    # vocab=10, tp=3 → num_embeddings_tp=4 → ranges (0,4),(4,4),(8,2)
    for r, expected in enumerate([(0, 4), (4, 4), (8, 2)]):
        _set_tp(r, 3)
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
        assert emb.vocab_range == expected, f"rank {r}: {emb.vocab_range} != {expected}"


def test_tp2_out_of_range_zero_before_all_reduce(_clean_kernel):
    """The value fed to all_reduce must have zeros on out-of-range positions."""
    from minisgl.layers.embedding import VocabParallelEmbedding

    captured = {}

    class _CapturingComm:
        def all_reduce(self, x):
            captured["y"] = x.detach().clone()
            return x
        def all_gather(self, x):
            return x

    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    _set_tp(0, 2)
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator", new=_CapturingComm
    ):
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb.weight = torch.nn.Parameter(_shard(W_full, 2, 0, emb.num_embeddings_tp))
    emb.forward(ids)
    y = captured["y"]
    # ids [0,4,5,9] on rank 0 → 5 and 9 are out of range; must be exactly zero
    assert torch.equal(y[2], torch.zeros(4))
    assert torch.equal(y[3], torch.zeros(4))
    # in-range unchanged
    assert torch.equal(y[0], W_full[0])
    assert torch.equal(y[1], W_full[4])


# ================================================================ 12. all_reduce only TP>1
def test_all_reduce_not_called_at_tp1(_tp1):
    from minisgl.layers.embedding import VocabParallelEmbedding
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator", new=_FakeComm
    ):
        emb = VocabParallelEmbedding(num_embeddings=32, embedding_dim=16)
    emb.weight = torch.nn.Parameter(torch.randn(32, 16))
    emb.forward(torch.tensor([0, 1, 2], dtype=torch.int32))
    assert emb._comm.all_reduce_calls == 0


# ================================================================ 13-14. CUDA
def _fake_cuda_int_tensor(list_):
    """Make a real CPU tensor but claim device.type == 'cuda' via a subclass wrap."""
    t = torch.tensor(list_, dtype=torch.int32)

    class _CudaFake(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:0")

    return t.as_subclass(_CudaFake)


def test_cuda_branch_calls_lazy_indexing_tp1(_tp1):
    """CUDA path: minisgl.kernel.indexing is imported inside forward and called."""
    from minisgl.layers.embedding import VocabParallelEmbedding
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator", new=_FakeComm
    ):
        emb = VocabParallelEmbedding(num_embeddings=32, embedding_dim=16)
    emb.weight = torch.nn.Parameter(torch.randn(32, 16))

    calls = []

    def _fake_indexing(*, weights, indices, vocab_range):
        calls.append({"weights_shape": tuple(weights.shape),
                      "indices_dtype": indices.dtype,
                      "vocab_range": vocab_range})
        return weights.new_zeros(indices.shape + (weights.shape[1],))

    fake_kernel = ModuleType("minisgl.kernel")
    fake_kernel.indexing = _fake_indexing
    with mock.patch.dict(sys.modules, {"minisgl.kernel": fake_kernel}):
        ids = _fake_cuda_int_tensor([0, 1, 2, 3])
        emb.forward(ids)

    assert len(calls) == 1
    assert calls[0]["vocab_range"] is None    # TP=1
    assert calls[0]["indices_dtype"] == torch.int32


def test_cuda_branch_passes_start_length_vocab_range_tp2():
    from minisgl.layers.embedding import VocabParallelEmbedding
    _set_tp(1, 2)
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator", new=_FakeComm
    ):
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb.weight = torch.nn.Parameter(torch.zeros(emb.num_embeddings_tp, 4))

    seen = []

    def _fake_indexing(*, weights, indices, vocab_range):
        seen.append(vocab_range)
        return weights.new_zeros(indices.shape + (weights.shape[1],))

    fake_kernel = ModuleType("minisgl.kernel")
    fake_kernel.indexing = _fake_indexing
    with mock.patch.dict(sys.modules, {"minisgl.kernel": fake_kernel}):
        ids = _fake_cuda_int_tensor([0, 4, 5, 9])
        emb.forward(ids)

    # rank=1, size=2, vocab=10, num_embeddings_tp=5 → (5,5)
    assert seen == [(5, 5)]


# ================================================================ 15. NPU CPU no import
def test_npu_cpu_path_never_imports_kernel(_clean_kernel):
    """Repeatedly exercise CPU forward (TP=1 and TP>1) — kernel stays out."""
    from minisgl.layers.embedding import VocabParallelEmbedding

    for m in list(sys.modules):
        if m == "minisgl.kernel" or m.startswith("minisgl.kernel."):
            sys.modules.pop(m, None)

    _set_tp(0, 1)
    emb1 = VocabParallelEmbedding(num_embeddings=32, embedding_dim=16)
    emb1.weight = torch.nn.Parameter(torch.randn(32, 16))
    emb1.forward(torch.tensor([0, 1], dtype=torch.int32))
    emb1.forward(torch.tensor([2, 3], dtype=torch.int64))

    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator", new=_FakeComm
    ):
        _set_tp(0, 2)
        emb2 = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb2.weight = torch.nn.Parameter(torch.randn(emb2.num_embeddings_tp, 4))
    emb2.forward(torch.tensor([0, 4, 5, 9], dtype=torch.int64))

    hits = [m for m in sys.modules if m == "minisgl.kernel" or m.startswith("minisgl.kernel.")]
    assert hits == [], f"NPU/CPU path leaked kernel import: {hits}"


# ================================================================ 16. source hygiene
def test_source_has_no_long_or_int64_coercion():
    src = EMB_SRC.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "long":
            pytest.fail(f".long attribute reference at line {node.lineno}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "to":
                for a in node.args:
                    if isinstance(a, ast.Attribute) and a.attr == "int64":
                        pytest.fail(f".to(torch.int64) at line {node.lineno}")
    assert ".long(" not in src
    assert "torch.int64" not in src


# ================================================================ 17. LMHead tied path unaffected
def test_parallel_lmhead_tied_state_dict_and_load(_tp1):
    from minisgl.layers.embedding import ParallelLMHead, VocabParallelEmbedding
    tied = VocabParallelEmbedding(num_embeddings=32, embedding_dim=16)
    tied.weight = torch.nn.Parameter(torch.randn(32, 16))
    head = ParallelLMHead(
        num_embeddings=32,
        embedding_dim=16,
        tie_word_embeddings=True,
        tied_embedding=tied,
    )
    # tied state_dict returns empty
    assert head.state_dict(prefix="lm_head") == {}

    # tied load pops lm_head.weight and .bias if present, otherwise no-op
    sd = {"lm_head.weight": torch.zeros(32, 16), "lm_head.bias": torch.zeros(32)}
    head.load_state_dict(sd, prefix="lm_head")
    assert "lm_head.weight" not in sd
    assert "lm_head.bias" not in sd

    # tied module still references the original weights
    assert head.tied_embedding is tied
    assert torch.equal(tied.weight, tied.weight)   # unchanged type


# ================================================================ 18. Gate 1.9l-fix
# The real DistributedCommunicator.all_reduce is *in-place* on the tensor; both
# TorchDistributedImpl (impl.py:26-31) and PyNCCLDistributedImpl (impl.py:48-50)
# return the same object they received. But the pynccl primitive itself is
# declared '-> None' (kernel/pynccl.py:18). To prove embedding never depends on
# the wrapper *returning* the tensor, we exercise it with a communicator that
# mutates in place and returns None.
class InplaceNoneCommunicator:
    """all_reduce mutates the tensor in place and returns None.

    This is the strictest form of the distributed-primitive contract: only the
    side effect on the argument is observable. Any embedding code that does
    ``return self._comm.all_reduce(y)`` breaks against this backend.
    """

    def __init__(self, factor=2.0):
        self.factor = factor
        self.all_reduce_calls = 0
        self.last_arg_id = None
        self.all_gather_calls = 0

    def all_reduce(self, tensor):
        self.all_reduce_calls += 1
        self.last_arg_id = id(tensor)
        tensor.mul_(self.factor)                 # in-place mutation, sentinel
        return None                              # explicit None contract

    def all_gather(self, tensor):
        self.all_gather_calls += 1
        return tensor


def test_tp2_forward_returns_tensor_not_none_under_inplace_none_comm(_clean_kernel):
    """Gate 1.9l-fix: TP>1 forward must NOT return whatever all_reduce returns.

    With a communicator whose all_reduce returns None, embedding must still
    return the tensor y that was passed to all_reduce.
    """
    from minisgl.layers.embedding import VocabParallelEmbedding
    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    _set_tp(0, 2)
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator",
        new=InplaceNoneCommunicator,
    ):
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb.weight = torch.nn.Parameter(_shard(W_full, 2, 0, emb.num_embeddings_tp))
    y = emb.forward(ids)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert emb._comm.all_reduce_calls == 1


def test_tp2_forward_returns_same_object_as_all_reduce_arg(_clean_kernel):
    """The returned tensor must BE the tensor that was passed to all_reduce
    (so the caller sees the in-place mutation)."""
    from minisgl.layers.embedding import VocabParallelEmbedding
    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    _set_tp(0, 2)
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator",
        new=InplaceNoneCommunicator,
    ):
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb.weight = torch.nn.Parameter(_shard(W_full, 2, 0, emb.num_embeddings_tp))
    y = emb.forward(ids)
    assert id(y) == emb._comm.last_arg_id


def test_tp2_inplace_mutation_by_all_reduce_visible_in_output(_clean_kernel):
    """The multiplication done inside all_reduce must be visible on the
    returned tensor — verifying the side effect propagates."""
    from minisgl.layers.embedding import VocabParallelEmbedding
    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    _set_tp(0, 2)
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator",
        new=InplaceNoneCommunicator,
    ):
        emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
    emb.weight = torch.nn.Parameter(_shard(W_full, 2, 0, emb.num_embeddings_tp))
    y = emb.forward(ids)
    # rank 0 owns [0,5) → ids [0,4,5,9] give rows W_full[0], W_full[4], 0, 0
    # then InplaceNoneCommunicator multiplied everything by factor=2.0.
    expected = torch.stack(
        [W_full[0] * 2.0, W_full[4] * 2.0, torch.zeros(4), torch.zeros(4)]
    )
    assert torch.equal(y, expected)


def test_tp1_does_not_call_all_reduce_under_inplace_none_comm(_clean_kernel, _tp1):
    """TP=1 must never touch all_reduce, even with a None-returning backend."""
    from minisgl.layers.embedding import VocabParallelEmbedding
    with mock.patch(
        "minisgl.layers.embedding.DistributedCommunicator",
        new=InplaceNoneCommunicator,
    ):
        emb = VocabParallelEmbedding(num_embeddings=32, embedding_dim=16)
    emb.weight = torch.nn.Parameter(torch.randn(32, 16))
    y = emb.forward(torch.tensor([0, 1, 2], dtype=torch.int32))
    assert y is not None
    assert emb._comm.all_reduce_calls == 0


def test_tp2_sum_matches_full_embedding_under_inplace_none_comm(_clean_kernel):
    """End-to-end sanity: with the in-place/None backend, do a manual per-rank
    sum and confirm we still recover W_full[ids] (factor=1.0 so mutation is
    a no-op equivalent to identity)."""
    from minisgl.layers.embedding import VocabParallelEmbedding
    W_full, ids = _tp2_setup(vocab=10, hidden=4)
    outs = []
    for r in range(2):
        _set_tp(r, 2)
        with mock.patch(
            "minisgl.layers.embedding.DistributedCommunicator",
            new=lambda: InplaceNoneCommunicator(factor=1.0),
        ):
            emb = VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)
        emb.weight = torch.nn.Parameter(_shard(W_full, 2, r, emb.num_embeddings_tp))
        y = emb.forward(ids)
        assert y is not None
        outs.append(y)
    total = outs[0] + outs[1]
    assert torch.equal(total, W_full[ids])


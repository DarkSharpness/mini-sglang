"""Gate 2.1c — SamplingParams.stop_token_ids field contract.

Pure-Python tests that never touch torch NPU / CUDA runtime, so they run on
any host in the pytest sweep. The invariants asserted here are the
dataclass-level guarantees the scheduler-side check relies on.
"""
from __future__ import annotations

import dataclasses

from minisgl.core import SamplingParams


def _sp_field(name: str) -> dataclasses.Field:
    return next(f for f in dataclasses.fields(SamplingParams) if f.name == name)


def test_stop_token_ids_field_declared() -> None:
    field_names = {f.name for f in dataclasses.fields(SamplingParams)}
    assert "stop_token_ids" in field_names, (
        "SamplingParams must declare a `stop_token_ids` field for Gate 2.1c"
    )


def test_stop_token_ids_default_is_empty_tuple() -> None:
    """Default MUST be the empty tuple — an immutable, safe-to-share sentinel."""
    field = _sp_field("stop_token_ids")
    # A mutable default (list) would be shared across instances and could not
    # even be declared without `field(default_factory=...)`. Assert the safe
    # tuple default is used.
    assert field.default == ()
    assert isinstance(field.default, tuple)
    # And there must be no default_factory (that would signal a mutable
    # workaround for a list default).
    assert field.default_factory is dataclasses.MISSING


def test_default_sampling_params_has_empty_stop_tokens() -> None:
    sp = SamplingParams()
    assert sp.stop_token_ids == ()
    assert isinstance(sp.stop_token_ids, tuple)


def test_two_default_instances_share_empty_tuple_by_identity() -> None:
    """The empty tuple is a Python singleton — sharing is safe because tuples
    are immutable. This test documents the property so a future contributor
    who accidentally switches to `list` will fail here."""
    sp1 = SamplingParams()
    sp2 = SamplingParams()
    assert sp1.stop_token_ids is sp2.stop_token_ids


def test_stop_token_ids_accepts_explicit_tuple() -> None:
    sp = SamplingParams(stop_token_ids=(11, 42))
    assert sp.stop_token_ids == (11, 42)
    assert 11 in sp.stop_token_ids
    assert 42 in sp.stop_token_ids
    assert 12 not in sp.stop_token_ids


def test_stop_token_ids_is_orthogonal_to_greedy() -> None:
    """Adding stop_token_ids must not perturb the greedy classifier."""
    sp = SamplingParams(stop_token_ids=(11,))
    assert sp.is_greedy is True
    sp2 = SamplingParams(temperature=0.7, stop_token_ids=(11,))
    assert sp2.is_greedy is False


def test_field_order_preserves_backward_kwargs() -> None:
    """Pre-Gate-2.1c positional users constructed SamplingParams with up to
    five positional args (temperature, top_k, top_p, ignore_eos, max_tokens).
    stop_token_ids must therefore appear AFTER those five to preserve the
    positional-argument contract."""
    fields = [f.name for f in dataclasses.fields(SamplingParams)]
    assert fields.index("stop_token_ids") > fields.index("max_tokens")
    assert fields.index("stop_token_ids") > fields.index("ignore_eos")

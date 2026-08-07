import sys
from types import SimpleNamespace

import torch
from minisgl.layers.rotary import _get_rope


def _stub_flashinfer(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(apply_rope_with_cos_sin_cache_inplace=lambda **kwargs: None),
    )


def test_yarn_attention_factor_scales_cos_sin_cache(monkeypatch):
    _stub_flashinfer(monkeypatch)
    attention_factor = 1.2079441541679836
    rope = _get_rope(
        head_dim=64,
        rotary_dim=64,
        max_position=16,
        base=500000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 8.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 8192,
            "attention_factor": attention_factor,
        },
    )

    cos, sin = rope._cos_sin_cache[0].chunk(2)
    assert torch.allclose(cos, torch.full_like(cos, attention_factor))
    assert torch.count_nonzero(sin) == 0


def test_default_rope_keeps_unit_attention_scaling(monkeypatch):
    _stub_flashinfer(monkeypatch)
    rope = _get_rope(
        head_dim=64,
        rotary_dim=64,
        max_position=16,
        base=500000.0,
        rope_scaling=None,
    )

    cos, sin = rope._cos_sin_cache[0].chunk(2)
    assert torch.equal(cos, torch.ones_like(cos))
    assert torch.count_nonzero(sin) == 0

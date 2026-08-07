from types import SimpleNamespace

import minisgl.layers.attention as attention_module
import pytest
import torch
from minisgl.layers.attention import AttentionLayer


class _RecordingNorm:
    def __init__(self):
        self.shape = None

    def forward_inplace(self, x):
        self.shape = tuple(x.shape)


class _IdentityRotary:
    def forward(self, positions, query, key):
        return query, key


class _AttentionBackend:
    def forward(self, q, k, v, layer_id, batch):
        return torch.zeros_like(q)


@pytest.mark.parametrize(
    ("mode", "q_shape", "k_shape"),
    [
        ("per_head", (3, 2, 4), (3, 1, 4)),
        ("projection", (3, 8), (3, 4)),
    ],
)
def test_qk_norm_mode_controls_normalization_axis(monkeypatch, mode, q_shape, k_shape):
    q_norm = _RecordingNorm()
    k_norm = _RecordingNorm()
    layer = object.__new__(AttentionLayer)
    layer.layer_id = 0
    layer.head_dim = 4
    layer.num_qo_heads = 2
    layer.num_kv_heads = 1
    layer.qo_attn_dim = 8
    layer.kv_attn_dim = 4
    layer.rotary = _IdentityRotary()
    layer.q_norm = q_norm
    layer.k_norm = k_norm
    layer.qk_norm_mode = mode

    context = SimpleNamespace(
        batch=SimpleNamespace(positions=torch.arange(3)),
        attn_backend=_AttentionBackend(),
    )
    monkeypatch.setattr(attention_module, "get_global_ctx", lambda: context)

    layer.forward(torch.randn(3, 16))

    assert q_norm.shape == q_shape
    assert k_norm.shape == k_shape

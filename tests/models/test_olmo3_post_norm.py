import torch
from minisgl.models.olmo3 import Olmo3DecoderLayer


class _Op:
    def __init__(self, fn):
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def test_olmo3_decoder_applies_norm_before_residual_add():
    layer = object.__new__(Olmo3DecoderLayer)
    layer.self_attn = _Op(lambda x: x + 1)
    layer.post_attention_layernorm = _Op(lambda x: x * 10)
    layer.mlp = _Op(lambda x: x + 2)
    layer.post_feedforward_layernorm = _Op(lambda x: x * 100)

    output = Olmo3DecoderLayer.forward.__wrapped__(layer, torch.tensor([2.0]))

    # Attention: (2 + 1) * 10 + 2 = 32
    # MLP:       (32 + 2) * 100 + 32 = 3432
    assert torch.equal(output, torch.tensor([3432.0]))

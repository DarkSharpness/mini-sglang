from types import SimpleNamespace

from minisgl.attention.fa import FlashAttentionBackend
from minisgl.attention.fi import FlashInferBackend


def _backend(layer_types, sliding_window=4096):
    backend = object.__new__(FlashAttentionBackend)
    backend.config = SimpleNamespace(
        model_type="olmo3",
        layer_types=layer_types,
        sliding_window=sliding_window,
    )
    return backend


def test_olmo3_attention_window_is_layer_aware():
    backend = _backend(("sliding_attention", "full_attention"))

    assert backend._window_size(0) == (4095, 0)
    assert backend._window_size(1) == (-1, -1)


def test_non_olmo_models_remain_full_attention():
    backend = object.__new__(FlashAttentionBackend)
    backend.config = SimpleNamespace(model_type="qwen3")

    assert backend._window_size(0) == (-1, -1)


def test_flashinfer_uses_the_same_olmo3_window_semantics():
    backend = object.__new__(FlashInferBackend)
    backend.config = SimpleNamespace(
        model_type="olmo3",
        layer_types=("sliding_attention", "full_attention"),
        sliding_window=4096,
    )

    assert backend._attention_type(0) == "sliding_attention"
    assert backend._attention_type(1) == "full_attention"
    assert backend._window_left("sliding_attention") == 4095
    assert backend._window_left("full_attention") == -1

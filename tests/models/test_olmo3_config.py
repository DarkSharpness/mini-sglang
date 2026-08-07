from types import SimpleNamespace

import pytest
from minisgl.models.config import ModelConfig


def _hf_config(**overrides):
    values = {
        "architectures": ["Olmo3ForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "max_position_embeddings": 65536,
        "model_type": "olmo3",
        "num_attention_heads": 32,
        "num_hidden_layers": 4,
        "num_key_value_heads": 32,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "attention_factor": 1.2079441541679836,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "yarn",
        },
        "rope_theta": 500000.0,
        "sliding_window": 4096,
        "tie_word_embeddings": False,
        "vocab_size": 100278,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_olmo3_layer_configuration_is_preserved():
    config = ModelConfig.from_hf(_hf_config())

    assert config.layer_types == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    )
    assert config.sliding_window == 4096
    assert config.rotary_config.scaling["attention_factor"] == pytest.approx(
        1.2079441541679836
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer_types": ["full_attention"]}, "one entry per hidden layer"),
        (
            {"layer_types": ["sliding_attention", "invalid", "full_attention", "full_attention"]},
            "Unsupported OLMo3 layer types",
        ),
        ({"sliding_window": 0}, "positive integer"),
    ],
)
def test_olmo3_invalid_layer_configuration_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_hf(_hf_config(**overrides))

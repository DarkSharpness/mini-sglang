from types import SimpleNamespace

import minisgl.utils.hf as hf_utils
from minisgl.llm.llm import LLM, RequestStatus
from minisgl.message import DetokenizeMsg
from minisgl.tokenizer.detokenize import DetokenizeManager


class _Tokenizer:
    eos_token_id = 100257

    def batch_decode(self, batches):
        return [" ".join(map(str, batch)) for batch in batches]


def test_generation_config_adds_all_eos_tokens(monkeypatch):
    monkeypatch.setattr(
        hf_utils.GenerationConfig,
        "from_pretrained",
        lambda _: SimpleNamespace(eos_token_id=[100265, 100257]),
    )

    assert hf_utils.load_eos_token_ids("model", _Tokenizer()) == frozenset(
        {100257, 100265}
    )


def test_generation_config_falls_back_to_tokenizer(monkeypatch):
    def _missing_generation_config(_):
        raise OSError("missing")

    monkeypatch.setattr(
        hf_utils.GenerationConfig, "from_pretrained", _missing_generation_config
    )
    monkeypatch.setattr(hf_utils, "cached_load_hf_config", lambda _: object())
    monkeypatch.setattr(
        hf_utils.GenerationConfig,
        "from_model_config",
        lambda _: (_ for _ in ()).throw(ValueError("missing")),
    )

    assert hf_utils.load_eos_token_ids("model", _Tokenizer()) == frozenset({100257})


def test_generation_config_falls_back_to_model_config(monkeypatch):
    monkeypatch.setattr(
        hf_utils.GenerationConfig,
        "from_pretrained",
        lambda _: (_ for _ in ()).throw(OSError("missing")),
    )
    monkeypatch.setattr(hf_utils, "cached_load_hf_config", lambda _: object())
    monkeypatch.setattr(
        hf_utils.GenerationConfig,
        "from_model_config",
        lambda _: SimpleNamespace(eos_token_id=[100265]),
    )

    assert hf_utils.load_eos_token_ids("model", _Tokenizer()) == frozenset(
        {100257, 100265}
    )


def test_detokenizer_filters_any_finished_eos_token():
    manager = DetokenizeManager(_Tokenizer(), {100257, 100265})

    output = manager.detokenize(
        [DetokenizeMsg(uid=1, next_token=100265, finished=True)]
    )

    assert output == [""]
    assert manager.decode_map == {}


def test_offline_llm_filters_secondary_finished_eos_token():
    llm = object.__new__(LLM)
    llm.eos_token_ids = frozenset({100257, 100265})
    llm.status_map = {1: RequestStatus(uid=1, input_ids=[1], output_ids=[])}

    llm.offline_send_result(
        [DetokenizeMsg(uid=1, next_token=100265, finished=True)]
    )

    assert llm.status_map[1].output_ids == []

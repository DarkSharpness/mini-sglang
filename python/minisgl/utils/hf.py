import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

_THINK_END_TOKENS = {
    # NOTE: register more reasoning model here
    "Qwen3ForCausalLM": "</think>",
    "Qwen3MoeForCausalLM": "</think>",
}


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    for arch in cached_load_hf_config(model_path).architectures:
        if arch not in _THINK_END_TOKENS:
            continue
        tokenizer.think_end_token = _THINK_END_TOKENS[arch]
        token_ids = tokenizer.encode(tokenizer.think_end_token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"{tokenizer.think_end_token!r} must map to exactly one token, got {token_ids}"
            )
        tokenizer.think_end_id = token_ids[0]
        break
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )

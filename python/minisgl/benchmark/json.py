from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_JSON_DATA_PATH = "NousResearch/json-mode-eval"


@dataclass(frozen=True)
class JsonBenchSample:
    system: str
    user: str
    json_schema: str

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    @property
    def response_format(self) -> dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {"schema": json.loads(self.json_schema)},
        }


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def iter_filtered_json_samples(
    data_path: str = DEFAULT_JSON_DATA_PATH,
) -> Iterator[JsonBenchSample]:
    from datasets import load_dataset

    raw_dataset = load_dataset(data_path)
    for data in raw_dataset["train"]:
        messages = data["prompt"]
        schema = json.loads(data["schema"])

        if schema.get("type") is None:
            continue

        assert len(messages) == 2, "invalid message length"
        system = messages[0]
        user = messages[1]
        assert system["role"] == "system", "invalid role"
        assert user["role"] == "user", "invalid role"
        yield JsonBenchSample(
            system=_normalize_content(system["content"]),
            user=_normalize_content(user["content"]),
            json_schema=json.dumps(schema, separators=(",", ":"), sort_keys=True),
        )


def collect_filtered_json_samples(
    n: int | None = None,
    data_path: str = DEFAULT_JSON_DATA_PATH,
) -> list[JsonBenchSample]:
    samples = []
    for sample in iter_filtered_json_samples(data_path):
        samples.append(sample)
        if n is not None and len(samples) >= n:
            break
    return samples


def collect_repeated_json_samples(
    n: int,
    data_path: str = DEFAULT_JSON_DATA_PATH,
) -> list[JsonBenchSample]:
    if n <= 0:
        return []

    samples = collect_filtered_json_samples(data_path=data_path)
    if not samples:
        return []
    return [samples[i % len(samples)] for i in range(n)]


def render_json_prompt_ids(
    tokenizer: PreTrainedTokenizerBase | Any,
    sample: JsonBenchSample,
) -> list[int]:
    return tokenizer.apply_chat_template(
        sample.messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def normalize_json_output(output: str) -> str:
    # Reasoning models may emit analysis before the final answer.
    if "</think>" in output:
        output = output.split("</think>", 1)[1]
    output = output.strip()

    # Some models wrap the final JSON in one fenced Markdown block.
    matched = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", output, re.S | re.I)
    if matched is not None:
        output = matched.group(1).strip()
    return output


def validate_json_output(output: str, json_schema: str) -> tuple[bool, bool | None]:
    output = normalize_json_output(output)

    try:
        obj = json.loads(output)
    except Exception:
        return False, False

    try:
        import jsonschema
    except ImportError:
        return True, None

    try:
        schema = json.loads(json_schema)
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())
        validator.validate(obj)
    except Exception:
        return True, False
    return True, True

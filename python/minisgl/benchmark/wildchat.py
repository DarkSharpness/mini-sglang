from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

LANGS = {"English", "Chinese"}
WILDCHAT_FIRST_SHARD = "train-00000-of-00086.parquet"
WILDCHAT_BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/"
BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmark"
DEFAULT_WILDCHAT_SHARD_PATH = BENCHMARK_DIR / "offline" / WILDCHAT_FIRST_SHARD


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    print(f"Downloading {url} -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp, path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def iter_filtered_wildchat_prompts(shard_path: Path = DEFAULT_WILDCHAT_SHARD_PATH):
    download_if_missing(WILDCHAT_BASE_URL + WILDCHAT_FIRST_SHARD, shard_path)

    parquet = pq.ParquetFile(shard_path)
    for batch in parquet.iter_batches(batch_size=256, columns=["conversation"]):
        prompts = []
        for conv in batch.to_pydict()["conversation"]:
            if not conv:
                continue

            first_user = None
            for turn in conv:
                if turn.get("role") == "user":
                    first_user = turn
                    break
            if first_user is None:
                continue

            text = (first_user.get("content") or "").strip()
            if not text:
                continue
            if first_user.get("language") not in LANGS:
                continue
            if bool(first_user.get("redacted")) or bool(first_user.get("toxic")):
                continue
            prompts.append(text)

        if not prompts:
            continue

        for prompt in prompts:
            yield prompt


def collect_filtered_wildchat_prompts(
    n: int | None = None,
    shard_path: Path = DEFAULT_WILDCHAT_SHARD_PATH,
) -> list[str]:
    prompts = []
    for prompt in iter_filtered_wildchat_prompts(shard_path):
        prompts.append(prompt)
        if n is not None and len(prompts) >= n:
            break
    return prompts


def iter_filtered_wildchat_prompt_ids(
    tokenizer: Any,
    shard_path: Path = DEFAULT_WILDCHAT_SHARD_PATH,
):
    for prompt in iter_filtered_wildchat_prompts(shard_path):
        yield tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )

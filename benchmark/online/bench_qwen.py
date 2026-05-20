from __future__ import annotations

import argparse
import asyncio
import os
import random
from pathlib import Path

from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.benchmark.json import validate_json_output
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


def download_qwen_trace(url: str) -> str:
    dir = Path(os.path.dirname(__file__))
    # download the file if not exists
    file_path = dir / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        import urllib.request

        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiniSGL with Qwen trace replay.")
    parser.add_argument(
        "--prompt-mode",
        choices=["dummy", "json"],
        default="dummy",
        help="Prompt source mode.",
    )
    parser.add_argument(
        "--json-mode",
        choices=["constrained", "unconstrained"],
        default="constrained",
        help="Only takes effect when --prompt-mode=json.",
    )
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Only takes effect when --prompt-mode=json.",
    )
    return parser.parse_args()


def process_json_correctness(results, traces) -> None:
    total = 0
    parse_ok = 0
    schema_ok = 0
    schema_checked = 0
    for trace, result in zip(traces, results, strict=True):
        if trace.json_schema is None:
            continue
        total += 1
        parsed, valid = validate_json_output(result.output_text, trace.json_schema)
        parse_ok += int(parsed)
        if valid is not None:
            schema_checked += 1
            schema_ok += int(valid)

    logger.info(f"JSON parse: {parse_ok}/{total}")
    logger.info(f"Schema valid: {schema_ok}/{schema_checked}")


async def main():
    args = parse_args()
    random.seed(42)  # reproducibility
    PORT = 1919
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        traces = read_qwen_trace(
            download_qwen_trace(URL),
            tokenizer,
            n=args.N,
            prompt_mode=args.prompt_mode,
            max_new_tokens=args.max_new_tokens,
            json_mode=args.json_mode,
        )

        logger.info(f"Start benchmarking with {len(traces)} requests using model {MODEL}...")

        for scale in SCALES:
            scaled_traces = scale_traces(traces, scale)
            results = await benchmark_trace(
                client,
                scaled_traces,
                MODEL,
            )
            process_benchmark_results(results)
            if args.prompt_mode == "json":
                process_json_correctness(results, scaled_traces)
        logger.info("Benchmarking completed.")


if __name__ == "__main__":
    asyncio.run(main())

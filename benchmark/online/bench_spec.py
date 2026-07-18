from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from typing import Any

from bench_spec_utils import (
    friendly_prompt,
    log_io_artifacts,
    log_server_artifact,
    maybe_wandb_run,
    mean_output_prompt_overlap,
    poll_spec_metrics,
    summarize,
    wandb_summary_payload,
)
from minisgl.benchmark.client import (
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from openai import AsyncOpenAI
from transformers import AutoTokenizer


async def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    async with AsyncOpenAI(
        base_url=f"http://127.0.0.1:{args.port}/v1", api_key="dummy"
    ) as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        if args.workload == "friendly":
            prompts = [
                friendly_prompt(tokenizer, args.input_len, i) for i in range(args.batch_size)
            ]
        else:
            prompts = [
                generate_prompt(tokenizer, args.input_len) for _ in range(args.batch_size)
            ]

        # Single fair knob between workloads: friendly disables Qwen3 thinking so the
        # model copies the document; adversarial keeps thinking on so it emits novel
        # reasoning tokens that the prompt-lookup drafter cannot match.
        enable_thinking = (
            args.enable_thinking
            if args.enable_thinking is not None
            else (args.workload == "adversarial")
        )
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

        config = {
            "model": model,
            "workload": args.workload,
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "seed": args.seed,
            "spec": args.spec,
            "overlap": args.overlap,
            "enable_thinking": enable_thinking,
            "revision": args.revision,
            "spec_num_draft": args.spec_num_draft,
            "spec_ngram_min": args.spec_ngram_min,
            "spec_ngram_max": args.spec_ngram_max,
        }
        print("BENCH_CONFIG " + json.dumps(config, sort_keys=True), flush=True)

        wb = maybe_wandb_run(config, enabled=args.wandb)
        try:
            warmup_bs = min(args.batch_size, 8)
            await benchmark_one_batch(
                client,
                prompts[:warmup_bs],
                min(args.output_len, 32),
                model,
                pbar=False,
                extra_body=extra_body,
            )

            # Single measured pass (like bench_qwen / bench_simple): one batch, no
            # repeat/aggregate, which also avoids re-sending identical prompts and
            # hitting the radix prefix cache across repeats.
            results = await benchmark_one_batch(
                client,
                prompts,
                args.output_len,
                model,
                pbar=False,
                extra_body=extra_body,
                capture_output=True,
            )
            process_benchmark_results(results)
            summary = summarize(results, args.output_len)
            overlap = mean_output_prompt_overlap(results, tokenizer, n=args.spec_ngram_max)

            final: dict[str, Any] = {
                **config,
                "duration_s": summary["duration_s"],
                "aggregate_output_tps": summary["aggregate_output_tps"],
                "mean_request_output_tps": summary["mean_request_output_tps"],
                "p50_request_output_tps": summary["p50_request_output_tps"],
            }
            if overlap is not None:
                final["output_prompt_overlap"] = overlap
            # Fresh server per cell (Modal) ⇒ process totals are this cell only.
            spec_metrics = (
                poll_spec_metrics(args.server_log)
                if args.spec and args.server_log
                else None
            )
            if spec_metrics is not None:
                final["spec_metrics"] = spec_metrics
                print("SPEC_METRICS " + json.dumps(spec_metrics, sort_keys=True), flush=True)
            elif args.spec and args.server_log:
                print("WARN: no SPEC_METRICS in server log after bench", flush=True)
            print("BENCH_RESULT " + json.dumps(final, sort_keys=True), flush=True)
            if wb is not None:
                payload = wandb_summary_payload(
                    final,
                    batch_wall_s=summary["duration_s"],
                    spec_metrics=spec_metrics,
                    output_prompt_overlap=overlap,
                )
                # One history row per cell creates the scalar bar charts; summary keeps
                # the same values for the run table.
                wb.log(payload)
                for key, value in payload.items():
                    wb.summary[key] = value
                log_io_artifacts(wb, results)
        finally:
            if wb is not None:
                log_server_artifact(wb, args.server_log)
                wb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["friendly", "adversarial"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=bool(os.environ.get("WANDB_API_KEY")),
        help="Live-log this cell to wandb (default: on when WANDB_API_KEY is set)",
    )
    parser.add_argument("--spec", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Qwen3 chat-template thinking mode; default off for friendly, on for adversarial",
    )
    parser.add_argument("--revision", default=os.environ.get("BENCH_REVISION", "unknown"))
    parser.add_argument(
        "--server-log",
        default="",
        help="Server log for SPEC_METRICS scrape + wandb artifact (empty disables)",
    )
    parser.add_argument("--spec-num-draft", type=int, default=4)
    parser.add_argument("--spec-ngram-min", type=int, default=1)
    parser.add_argument("--spec-ngram-max", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.input_len, args.output_len) < 1:
        parser.error("batch-size, input-len, and output-len must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
